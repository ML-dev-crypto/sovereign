#include "sovereign/revised_simplex.hpp"

#include "sovereign/dense_lu.hpp"
#include "sovereign/sparse_matrix.hpp"
#include "sovereign/tolerances.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace sovereign {
namespace {

struct StandardLp {
  SparseMatrixCSC A;                 // m x n equalities
  std::vector<double> b;             // m
  std::vector<double> c;             // n (minimization)
  std::vector<std::string> names;    // original structural names; empty for aux
  std::vector<int> structural_index; // maps col -> original var index, or -1
  std::vector<double> shift;         // original x = shift + y for structural
  std::vector<double> col_scale;     // x_unscaled = col_scale * x_scaled
  std::vector<int> logical_basis;    // size m: preferred initial basis column per row
  int n_structural = 0;
  int m = 0;
  int n = 0;
  Sense original_sense = Sense::Minimize;
  std::vector<std::string> warnings;
};

double clamp_nonnegative(double v, double tol) {
  return (v < 0.0 && v > -tol) ? 0.0 : v;
}

StandardLp build_standard_form(const OptimizationModel& model, bool enable_scaling) {
  StandardLp lp;
  lp.original_sense = model.sense;
  lp.n_structural = static_cast<int>(model.variables.size());
  lp.m = static_cast<int>(model.constraints.size());

  // Shifted structural variables y_i = x_i - lb_i >= 0
  // Finite upper bounds become extra <= constraints.
  struct BoundConstraint {
    int var = 0;
    double ub = 0.0;
  };
  std::vector<BoundConstraint> ub_cons;
  lp.shift.assign(static_cast<std::size_t>(lp.n_structural), 0.0);

  for (int i = 0; i < lp.n_structural; ++i) {
    const Variable& v = model.variables[static_cast<std::size_t>(i)];
    lp.shift[static_cast<std::size_t>(i)] = v.lower_bound;
    if (std::isfinite(v.upper_bound) && v.upper_bound < 1e29) {
      BoundConstraint bc;
      bc.var = i;
      bc.ub = v.upper_bound - v.lower_bound;
      if (bc.ub < -1e-12) {
        throw std::runtime_error("Inconsistent bounds for variable " + v.name);
      }
      ub_cons.push_back(bc);
    }
  }

  const int m_total = lp.m + static_cast<int>(ub_cons.size());
  // Count auxiliary columns: slack/surplus per original constraint + slack per ub
  // Equality: surplus=0, need artificial later
  // We'll create: for each <= : slack; for each >= : surplus; for each = : none yet
  // plus ub slacks.

  std::vector<double> rhs(static_cast<std::size_t>(m_total), 0.0);
  std::vector<std::vector<std::pair<int, double>>> rows(
      static_cast<std::size_t>(m_total));

  auto add_coeff = [&](int row, int col, double val) {
    if (val == 0.0) return;
    rows[static_cast<std::size_t>(row)].push_back({col, val});
  };

  // Structural columns 0..n_structural-1
  std::unordered_map<std::string, int> var_index;
  var_index.reserve(static_cast<std::size_t>(lp.n_structural) * 2);
  for (int i = 0; i < lp.n_structural; ++i) {
    var_index[model.variables[static_cast<std::size_t>(i)].name] = i;
  }
  for (int r = 0; r < lp.m; ++r) {
    const Constraint& cons = model.constraints[static_cast<std::size_t>(r)];
    double b = cons.rhs;
    // Account for shifts: a'(lb + y) ? rhs => a'y ? rhs - a'lb
    for (const auto& kv : cons.linear) {
      auto it = var_index.find(kv.first);
      if (it == var_index.end()) {
        throw std::runtime_error("Unknown variable in constraint: " + kv.first);
      }
      const int j = it->second;
      add_coeff(r, j, kv.second);
      b -= kv.second * lp.shift[static_cast<std::size_t>(j)];
    }
    rhs[static_cast<std::size_t>(r)] = b;
  }

  for (std::size_t k = 0; k < ub_cons.size(); ++k) {
    const int row = lp.m + static_cast<int>(k);
    add_coeff(row, ub_cons[k].var, 1.0);
    rhs[static_cast<std::size_t>(row)] = ub_cons[k].ub;
  }

  // Allocate auxiliary columns while building CSC.
  // Column layout: [structural | slacks/surplus | artificials]
  // First pass: determine aux for constraints.
  enum class RowKind { Le, Ge, Eq };
  std::vector<RowKind> kinds(static_cast<std::size_t>(m_total), RowKind::Le);
  for (int r = 0; r < lp.m; ++r) {
    const auto s = model.constraints[static_cast<std::size_t>(r)].sense;
    if (s == ConstraintSense::Le) kinds[static_cast<std::size_t>(r)] = RowKind::Le;
    else if (s == ConstraintSense::Ge) kinds[static_cast<std::size_t>(r)] = RowKind::Ge;
    else kinds[static_cast<std::size_t>(r)] = RowKind::Eq;
  }
  for (std::size_t k = 0; k < ub_cons.size(); ++k) {
    kinds[static_cast<std::size_t>(lp.m + static_cast<int>(k))] = RowKind::Le;
  }

  // Flip rows with negative RHS to keep b >= 0 where helpful (except we allow neg b
  // and use artificials). For Ge with positive b after shift ok.
  // If RHS < 0, multiply row by -1 and flip sense.
  for (int r = 0; r < m_total; ++r) {
    if (rhs[static_cast<std::size_t>(r)] < 0.0) {
      rhs[static_cast<std::size_t>(r)] = -rhs[static_cast<std::size_t>(r)];
      for (auto& e : rows[static_cast<std::size_t>(r)]) e.second = -e.second;
      if (kinds[static_cast<std::size_t>(r)] == RowKind::Le)
        kinds[static_cast<std::size_t>(r)] = RowKind::Ge;
      else if (kinds[static_cast<std::size_t>(r)] == RowKind::Ge)
        kinds[static_cast<std::size_t>(r)] = RowKind::Le;
      // Eq stays Eq
    }
  }

  int n_slack = 0;
  for (int r = 0; r < m_total; ++r) {
    if (kinds[static_cast<std::size_t>(r)] == RowKind::Le) ++n_slack;
    if (kinds[static_cast<std::size_t>(r)] == RowKind::Ge) ++n_slack;  // surplus
  }

  // Artificials: for Ge and Eq (and Le if we want identity start - Le has slack)
  // Phase I needs artificial on Ge and Eq. For Le, slack forms identity basis.
  int n_art = 0;
  for (int r = 0; r < m_total; ++r) {
    if (kinds[static_cast<std::size_t>(r)] != RowKind::Le) ++n_art;
  }

  const int n_total = lp.n_structural + n_slack + n_art;
  lp.n = n_total;
  lp.m = m_total;
  lp.A.resize(static_cast<std::size_t>(m_total), static_cast<std::size_t>(n_total));
  lp.A.reserve_nnz(static_cast<std::size_t>(n_total * 2));
  lp.b = rhs;
  lp.c.assign(static_cast<std::size_t>(n_total), 0.0);
  lp.names.assign(static_cast<std::size_t>(n_total), std::string());
  lp.structural_index.assign(static_cast<std::size_t>(n_total), -1);

  // Objective on structural (minimization). Maximize => negate.
  const double sense_sign = (model.sense == Sense::Maximize) ? -1.0 : 1.0;
  for (int i = 0; i < lp.n_structural; ++i) {
    lp.names[static_cast<std::size_t>(i)] = model.variables[static_cast<std::size_t>(i)].name;
    lp.structural_index[static_cast<std::size_t>(i)] = i;
    double coef = 0.0;
    auto it = model.objective.linear.find(lp.names[static_cast<std::size_t>(i)]);
    if (it != model.objective.linear.end()) coef = it->second;
    lp.c[static_cast<std::size_t>(i)] = sense_sign * coef;
  }

  // Build columns: first collect triplets then compress by column
  std::vector<std::vector<std::pair<int, double>>> cols(
      static_cast<std::size_t>(n_total));

  for (int r = 0; r < m_total; ++r) {
    for (const auto& e : rows[static_cast<std::size_t>(r)]) {
      cols[static_cast<std::size_t>(e.first)].push_back({r, e.second});
    }
  }

  int slack_col = lp.n_structural;
  int art_col = lp.n_structural + n_slack;
  lp.logical_basis.assign(static_cast<std::size_t>(m_total), -1);

  for (int r = 0; r < m_total; ++r) {
    if (kinds[static_cast<std::size_t>(r)] == RowKind::Le) {
      cols[static_cast<std::size_t>(slack_col)].push_back({r, 1.0});
      lp.names[static_cast<std::size_t>(slack_col)] = "__slack_" + std::to_string(r);
      lp.logical_basis[static_cast<std::size_t>(r)] = slack_col;
      ++slack_col;
    } else if (kinds[static_cast<std::size_t>(r)] == RowKind::Ge) {
      cols[static_cast<std::size_t>(slack_col)].push_back({r, -1.0});
      lp.names[static_cast<std::size_t>(slack_col)] = "__surplus_" + std::to_string(r);
      ++slack_col;
      cols[static_cast<std::size_t>(art_col)].push_back({r, 1.0});
      lp.names[static_cast<std::size_t>(art_col)] = "__art_" + std::to_string(r);
      lp.logical_basis[static_cast<std::size_t>(r)] = art_col;
      ++art_col;
    } else {
      cols[static_cast<std::size_t>(art_col)].push_back({r, 1.0});
      lp.names[static_cast<std::size_t>(art_col)] = "__art_" + std::to_string(r);
      lp.logical_basis[static_cast<std::size_t>(r)] = art_col;
      ++art_col;
    }
  }

  for (int j = 0; j < n_total; ++j) {
    auto& entries = cols[static_cast<std::size_t>(j)];
    std::sort(entries.begin(), entries.end(),
              [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                return a.first < b.first;
              });
    // merge duplicates
    std::vector<std::pair<int, double>> merged;
    for (const auto& e : entries) {
      if (!merged.empty() && merged.back().first == e.first) {
        merged.back().second += e.second;
      } else {
        merged.push_back(e);
      }
    }
    for (const auto& e : merged) {
      if (std::abs(e.second) > 0.0) {
        lp.A.push_back(e.first, e.second);
      }
    }
    lp.A.finish_column(static_cast<std::size_t>(j));
  }

  // Scaling: geometric row/column equilibration (CSC-friendly, O(nnz) per pass)
  lp.col_scale.assign(static_cast<std::size_t>(n_total), 1.0);
  if (enable_scaling && m_total > 0 && n_total > 0) {
    for (int pass = 0; pass < 2; ++pass) {
      std::vector<double> row_max(static_cast<std::size_t>(m_total), 0.0);
      for (std::size_t p = 0; p < lp.A.values.size(); ++p) {
        const int r = lp.A.row_idx[p];
        row_max[static_cast<std::size_t>(r)] =
            std::max(row_max[static_cast<std::size_t>(r)], std::abs(lp.A.values[p]));
      }
      std::vector<double> row_scale(static_cast<std::size_t>(m_total), 1.0);
      for (int i = 0; i < m_total; ++i) {
        if (row_max[static_cast<std::size_t>(i)] > 0.0) {
          row_scale[static_cast<std::size_t>(i)] =
              1.0 / std::sqrt(row_max[static_cast<std::size_t>(i)]);
        }
      }
      for (int i = 0; i < m_total; ++i) {
        lp.b[static_cast<std::size_t>(i)] *= row_scale[static_cast<std::size_t>(i)];
      }
      for (std::size_t p = 0; p < lp.A.values.size(); ++p) {
        lp.A.values[p] *= row_scale[static_cast<std::size_t>(lp.A.row_idx[p])];
      }

      std::vector<double> s(static_cast<std::size_t>(n_total), 1.0);
      for (int j = 0; j < n_total; ++j) {
        double maxv = 0.0;
        for (int p = lp.A.col_ptr[static_cast<std::size_t>(j)];
             p < lp.A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
          maxv = std::max(maxv, std::abs(lp.A.values[static_cast<std::size_t>(p)]));
        }
        if (maxv > 0.0) s[static_cast<std::size_t>(j)] = 1.0 / std::sqrt(maxv);
      }
      for (int j = 0; j < n_total; ++j) {
        for (int p = lp.A.col_ptr[static_cast<std::size_t>(j)];
             p < lp.A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
          lp.A.values[static_cast<std::size_t>(p)] *= s[static_cast<std::size_t>(j)];
        }
        lp.c[static_cast<std::size_t>(j)] *= s[static_cast<std::size_t>(j)];
        lp.col_scale[static_cast<std::size_t>(j)] *= s[static_cast<std::size_t>(j)];
      }
    }
    lp.warnings.push_back("Row/column equilibration scaling applied.");
  }

  // Stash done via logical_basis
  return lp;
}

bool is_artificial(const std::string& name) {
  return name.compare(0, 6, "__art_") == 0;
}

struct Eta {
  int pivot = -1;              // leaving row position
  std::vector<double> alpha;   // B_old^{-1} a_enter at pivot time
};

struct SimplexState {
  const StandardLp* lp = nullptr;
  Tolerances tol;
  std::vector<int> basis;     // size m, column indices
  std::vector<int> nonbasic;  // size n-m
  std::vector<char> is_basic; // size n
  std::vector<double> xB;
  DenseLU lu;                 // factorization of basis at last refactor
  std::vector<Eta> etas;      // product-form updates since last refactor
  std::int64_t iterations = 0;
  int since_refactor = 0;
  int refactor_every = 64;
};

void apply_eta_inv(std::vector<double>& x, int p, const std::vector<double>& alpha) {
  const double piv = alpha[static_cast<std::size_t>(p)];
  const double xp = x[static_cast<std::size_t>(p)] / piv;
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (static_cast<int>(i) == p) continue;
    x[i] -= alpha[i] * xp;
  }
  x[static_cast<std::size_t>(p)] = xp;
}

void apply_eta_inv_transpose(std::vector<double>& x, int p,
                             const std::vector<double>& alpha) {
  // E has column p equal to `alpha` (alpha[p] is the pivot element).
  // E^{-1} row p is (1/alpha_p) at column p; row i!=p is 1 at column i and
  // -alpha_i/alpha_p at column p. Transposing swaps rows/cols, so:
  //   (E^{-T} x)_i = x_i                                  for i != p
  //   (E^{-T} x)_p = (x_p - sum_{i!=p} alpha_i * x_i) / alpha_p
  // The previous implementation omitted the division of x_p itself by
  // alpha_p, which corrupted the dual vector (pi) computed by btran
  // whenever any eta's pivot element was not exactly 1 — producing wrong
  // reduced costs and causing the simplex to terminate at a dual-infeasible
  // "optimal" basis (converging to a suboptimal vertex without any further
  // detected improving column).
  const double piv = alpha[static_cast<std::size_t>(p)];
  double sum = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (static_cast<int>(i) == p) continue;
    sum += alpha[i] * x[i];
  }
  x[static_cast<std::size_t>(p)] = (x[static_cast<std::size_t>(p)] - sum) / piv;
}

bool ftran(SimplexState& st, std::vector<double>& x) {
  // Solve B x = a with B = B0 * E1 * ... * Ek
  if (!st.lu.solve(x)) return false;
  for (const Eta& e : st.etas) {
    apply_eta_inv(x, e.pivot, e.alpha);
  }
  return true;
}

bool btran(SimplexState& st, std::vector<double>& x) {
  // Solve B^T pi = c, where B = B0 * E1 * E2 * ... * Ek (product-form basis).
  // B^T = Ek^T * ... * E1^T * B0^T, so B^{-T} = B0^{-T} * E1^{-T} * ... * Ek^{-T}.
  // The base factorization's transpose-solve must be applied FIRST, then the
  // eta transposes in forward (insertion) order. Applying them in reverse
  // order (as a prior version of this function did) computes the wrong dual
  // vector whenever any etas are present, which corrupts reduced-cost
  // pricing and can cause the simplex to genuinely cycle forever (observed:
  // repeating exact 2-cycles for 100k+ iterations on degenerate MILP node
  // LPs) because entering-variable choices are based on incorrect duals.
  if (!st.lu.solve_transpose(x)) return false;
  for (const Eta& e : st.etas) {
    apply_eta_inv_transpose(x, e.pivot, e.alpha);
  }
  return true;
}

bool refactor_basis(SimplexState& st) {
  std::vector<double> dense;
  st.lp->A.extract_dense_basis(st.basis, dense);
  if (!st.lu.factorize(std::move(dense), static_cast<std::size_t>(st.lp->m))) {
    return false;
  }
  st.etas.clear();
  st.since_refactor = 0;
  return true;
}

bool compute_xB(SimplexState& st) {
  st.xB = st.lp->b;
  if (!ftran(st, st.xB)) return false;
  for (double& v : st.xB) {
    if (std::abs(v) < st.tol.zero) v = 0.0;
  }
  return true;
}

void rebuild_nonbasic(SimplexState& st) {
  st.nonbasic.clear();
  for (int j = 0; j < st.lp->n; ++j) {
    if (!st.is_basic[static_cast<std::size_t>(j)]) {
      st.nonbasic.push_back(j);
    }
  }
}

bool initial_basis(SimplexState& st) {
  const int m = st.lp->m;
  const int n = st.lp->n;
  st.basis.assign(static_cast<std::size_t>(m), -1);
  st.is_basic.assign(static_cast<std::size_t>(n), 0);

  if (static_cast<int>(st.lp->logical_basis.size()) != m) {
    return false;
  }
  for (int i = 0; i < m; ++i) {
    const int j = st.lp->logical_basis[static_cast<std::size_t>(i)];
    if (j < 0 || j >= n) return false;
    if (st.is_basic[static_cast<std::size_t>(j)]) return false;
    st.basis[static_cast<std::size_t>(i)] = j;
    st.is_basic[static_cast<std::size_t>(j)] = 1;
  }
  rebuild_nonbasic(st);
  if (!refactor_basis(st)) return false;
  return compute_xB(st);
}

// Phase I costs: 1 for artificials, 0 else. Phase II: original c (artificials fixed 0).
void reduced_costs(SimplexState& st, const std::vector<double>& c,
                   std::vector<double>& rc) {
  // pi^T = c_B^T B^{-1}  => solve B^T pi = c_B
  std::vector<double> cB(static_cast<std::size_t>(st.lp->m), 0.0);
  for (int i = 0; i < st.lp->m; ++i) {
    cB[static_cast<std::size_t>(i)] = c[static_cast<std::size_t>(st.basis[static_cast<std::size_t>(i)])];
  }
  std::vector<double> pi = cB;
  btran(st, pi);

  rc.assign(static_cast<std::size_t>(st.lp->n), 0.0);
  for (int j : st.nonbasic) {
    double aj_pi = 0.0;
    for (int p = st.lp->A.col_ptr[static_cast<std::size_t>(j)];
         p < st.lp->A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
      aj_pi += st.lp->A.values[static_cast<std::size_t>(p)] *
               pi[static_cast<std::size_t>(st.lp->A.row_idx[static_cast<std::size_t>(p)])];
    }
    rc[static_cast<std::size_t>(j)] = c[static_cast<std::size_t>(j)] - aj_pi;
  }
}

int select_entering(const SimplexState& st, const std::vector<double>& rc, bool bland,
                    bool allow_artificials) {
  int enter = -1;
  double best = -st.tol.optimality;
  for (int j : st.nonbasic) {
    if (!allow_artificials && is_artificial(st.lp->names[static_cast<std::size_t>(j)])) {
      continue;
    }
    const double rj = rc[static_cast<std::size_t>(j)];
    if (rj < -st.tol.optimality) {
      if (bland) {
        if (enter < 0 || j < enter) enter = j;
      } else if (rj < best) {
        best = rj;
        enter = j;
      }
    }
  }
  return enter;
}

bool pivot_column(SimplexState& st, int enter, std::vector<double>& d) {
  st.lp->A.extract_column(static_cast<std::size_t>(enter), d);
  return ftran(st, d);
}

int select_leaving(const SimplexState& st, const std::vector<double>& d, bool bland) {
  // Harris ratio test: Instead of selecting the single minimum ratio, collect
  // ALL rows where ratio is within harris_tol of minimum, then among those
  // candidates select the one with largest |d[i]| (pivot magnitude).
  // This avoids small pivots that cause numerical instability and cycling.
  //
  // When bland is set, use Bland's rule (smallest variable index) instead of
  // largest pivot among candidates.

  // First pass: find minimum ratio
  double min_ratio = std::numeric_limits<double>::infinity();
  for (int i = 0; i < st.lp->m; ++i) {
    const double di = d[static_cast<std::size_t>(i)];
    if (di <= st.tol.pivot) continue;
    const double ratio = st.xB[static_cast<std::size_t>(i)] / di;
    if (ratio < min_ratio) min_ratio = ratio;
  }

  if (std::isinf(min_ratio)) return -1;  // unbounded

  // Second pass: Harris selection among near-minimum candidates
  const double harris_tol = st.tol.feasibility * std::max(1.0, std::abs(min_ratio));
  int leave_pos = -1;
  double best_pivot = 0.0;
  int best_var = std::numeric_limits<int>::max();

  for (int i = 0; i < st.lp->m; ++i) {
    const double di = d[static_cast<std::size_t>(i)];
    if (di <= st.tol.pivot) continue;
    const double ratio = st.xB[static_cast<std::size_t>(i)] / di;

    if (ratio <= min_ratio + harris_tol) {
      const int var = st.basis[static_cast<std::size_t>(i)];
      if (bland) {
        // Bland: smallest variable index among candidates
        if (leave_pos < 0 || var < best_var) {
          leave_pos = i;
          best_var = var;
          best_pivot = di;
        }
      } else {
        // Harris: largest pivot magnitude among candidates
        if (std::abs(di) > best_pivot) {
          best_pivot = std::abs(di);
          leave_pos = i;
          best_var = var;
        }
      }
    }
  }
  return leave_pos;
}

bool perform_pivot(SimplexState& st, int enter, int leave_pos,
                   const std::vector<double>& d) {
  const double d_piv = d[static_cast<std::size_t>(leave_pos)];
  if (std::abs(d_piv) < st.tol.pivot) return false;

  // Update primal basic solution: xB <- xB - theta * d, xB[leave]=theta
  const double theta = st.xB[static_cast<std::size_t>(leave_pos)] / d_piv;
  for (int i = 0; i < st.lp->m; ++i) {
    st.xB[static_cast<std::size_t>(i)] -= theta * d[static_cast<std::size_t>(i)];
  }
  st.xB[static_cast<std::size_t>(leave_pos)] = theta;

  const int leave = st.basis[static_cast<std::size_t>(leave_pos)];
  st.is_basic[static_cast<std::size_t>(leave)] = 0;
  st.is_basic[static_cast<std::size_t>(enter)] = 1;
  st.basis[static_cast<std::size_t>(leave_pos)] = enter;

  // O(1)-ish nonbasic update: replace entering slot with leaving column
  bool swapped = false;
  for (int& j : st.nonbasic) {
    if (j == enter) {
      j = leave;
      swapped = true;
      break;
    }
  }
  if (!swapped) {
    rebuild_nonbasic(st);
  }

  // Product-form update: B_new = B_old * E  (column leave_pos of E is d)
  Eta eta;
  eta.pivot = leave_pos;
  eta.alpha = d;
  st.etas.push_back(std::move(eta));
  ++st.since_refactor;

  // Periodic refactor for numerical stability (NOT every pivot)
  if (st.since_refactor >= st.refactor_every) {
    if (!refactor_basis(st)) return false;
    if (!compute_xB(st)) return false;
  }
  return true;
}

enum class PhaseStatus { Optimal, Unbounded, Singular, IterationLimit };

// A phase that stops early is not an answer. Each terminal reason maps to the
// status the caller actually sees, so "we ran out of iterations" is never
// reported as a generic ERROR (which reads like a bug) nor as OPTIMAL.
SolverStatus status_for(PhaseStatus ps) {
  switch (ps) {
    case PhaseStatus::Singular:
      return SolverStatus::NumericalError;
    case PhaseStatus::IterationLimit:
      return SolverStatus::IterationLimit;
    default:
      return SolverStatus::Error;
  }
}

PhaseStatus run_phase(SimplexState& st, const std::vector<double>& c,
                      int max_iterations, bool allow_artificials, std::string* detail) {
  // Sticky Bland is LOCAL to this phase/call — a fresh stack bool, never stored
  // on SimplexState or any object reused across MILP node LPs. Each
  // RevisedSimplexSolver::solve builds a new SimplexState, so Bland cannot
  // leak across nodes. Once a degenerate pivot is seen, Bland stays on for
  // the rest of THIS phase only (Phase I / Phase II each get a fresh flag).
  //
  // Under Bland we also drop product-form etas before pricing (refactor).
  // Lex RHS perturbation alone is not always enough to stop exact 2-cycles
  // when duals drift through a long eta chain; the combo of lex + Bland
  // selection + clean factor under Bland is what terminates the dumped
  // set-partition/knapsack node LPs. Cost is paid only after degeneracy is
  // detected, not on every non-degenerate large LP from the start.
  bool use_bland = false;
  int improving_streak = 0;
  const bool log_cycle = std::getenv("SOVEREIGN_DBG_CYCLE") != nullptr;

  // Cycle detection buffer: track last 10 (enter, leave) pairs
  // If we see the same pair repeat within this window, force refactorization
  // to clear accumulated numerical error in the eta chain
  struct PivotPair {
    int enter = -1;
    int leave = -1;
  };
  std::vector<PivotPair> recent_pivots;
  recent_pivots.reserve(10);

  while (st.iterations < max_iterations) {
    // Under Bland, drop product-form etas before pricing so duals cannot
    // drift through a long eta chain (the observed 6↔8 exact 2-cycle).
    // This is O(m³) per pivot while Bland is active — acceptable on small
    // degenerate MILP node LPs; large LPs un-stick after a run of improving
    // pivots (below) so they do not stay on this path for the whole solve.
    if (use_bland && !st.etas.empty()) {
      if (!refactor_basis(st) || !compute_xB(st)) {
        if (detail) *detail = "Basis refactorization failed under Bland mode.";
        return PhaseStatus::Singular;
      }
    }

    std::vector<double> rc;
    reduced_costs(st, c, rc);
    int enter = select_entering(st, rc, use_bland, allow_artificials);
    if (enter < 0) {
      return PhaseStatus::Optimal;
    }
    std::vector<double> d;
    if (!pivot_column(st, enter, d)) {
      if (!refactor_basis(st) || !compute_xB(st) || !pivot_column(st, enter, d)) {
        if (detail) *detail = "Basis solve failed for entering column.";
        return PhaseStatus::Singular;
      }
    }
    int leave_pos = select_leaving(st, d, use_bland);
    if (leave_pos < 0) {
      return PhaseStatus::Unbounded;
    }
    const int leave_var = st.basis[static_cast<std::size_t>(leave_pos)];
    const double ratio =
        st.xB[static_cast<std::size_t>(leave_pos)] / d[static_cast<std::size_t>(leave_pos)];

    // PART 1: Cycle detection - check if this (enter, leave_var) pair was seen recently
    bool cycle_detected = false;
    for (const auto& pp : recent_pivots) {
      if (pp.enter == enter && pp.leave == leave_var) {
        cycle_detected = true;
        break;
      }
    }

    if (cycle_detected) {
      // Force refactorization to clear accumulated numerical error
      if (log_cycle) {
        std::cerr << "[cycle] DETECTED at iteration " << st.iterations
                  << " enter=" << enter << " leave=" << leave_var
                  << " - forcing refactorization\n";
      }
      if (!refactor_basis(st) || !compute_xB(st)) {
        if (detail) *detail = "Refactorization failed after cycle detection.";
        return PhaseStatus::Singular;
      }
      recent_pivots.clear();  // Clear history after refactor
      use_bland = true;       // Switch to Bland's rule to help break the cycle
      improving_streak = 0;
    } else {
      // Record this pivot
      PivotPair pp;
      pp.enter = enter;
      pp.leave = leave_var;
      recent_pivots.push_back(pp);
      if (recent_pivots.size() > 10) {
        recent_pivots.erase(recent_pivots.begin());  // Keep only last 10
      }
    }

    if (ratio <= st.tol.feasibility) {
      use_bland = true;
      improving_streak = 0;
    } else if (use_bland) {
      ++improving_streak;
      // Leave the slow Bland+refactor path once we are making real progress
      // again. Lex perturbation remains active for the whole solve, so
      // re-entering Bland later if degeneracy returns is safe.
      if (improving_streak >= 64) {
        use_bland = false;
        improving_streak = 0;
      }
    }
    if (log_cycle && st.iterations < 200) {
      std::cerr << "[cycle] it=" << st.iterations << " bland=" << use_bland
                << " enter=" << enter << " leave=" << leave_var
                << " leave_row=" << leave_pos << " ratio=" << ratio << "\n";
    }

    if (!perform_pivot(st, enter, leave_pos, d)) {
      if (detail) *detail = "Basis update / refactorization failed.";
      return PhaseStatus::Singular;
    }
    ++st.iterations;

    for (double& v : st.xB) {
      if (v < 0.0 && v > -st.tol.feasibility) v = 0.0;
    }
  }
  if (detail) *detail = "Iteration limit reached.";
  return PhaseStatus::IterationLimit;
}

SolverResult solve_standard(const StandardLp& lp, const RevisedSimplexOptions& opt,
                            const OptimizationModel& original) {
  SolverResult result;
  result.warnings = lp.warnings;

  if (lp.m == 0) {
    // Unconstrained besides bounds already encoded; minimize c'y with y>=0
    // If any c_j < 0 unbounded; else y=0.
    bool unbounded = false;
    for (int j = 0; j < lp.n_structural; ++j) {
      if (lp.c[static_cast<std::size_t>(j)] < -opt.optimality_tol) unbounded = true;
    }
    if (unbounded) {
      result.status = SolverStatus::Unbounded;
      result.message = "Problem is unbounded.";
      return result;
    }
    result.status = SolverStatus::Optimal;
    result.has_objective_value = true;
    double obj = original.objective.constant;
    for (int i = 0; i < lp.n_structural; ++i) {
      const double x = lp.shift[static_cast<std::size_t>(i)];
      result.primal[lp.names[static_cast<std::size_t>(i)]] = x;
      auto it = original.objective.linear.find(lp.names[static_cast<std::size_t>(i)]);
      if (it != original.objective.linear.end()) obj += it->second * x;
    }
    result.objective_value = obj;
    result.message = "Optimal (no constraints).";
    return result;
  }

  // Lexicographic RHS perturbation (practical anti-cycling).
  // Only perturb rows whose initial logical basis is a real slack (not an
  // artificial). Perturbing equality / >= rows that start with an artificial
  // shifts a structurally tight RHS and can make Phase I report false
  // infeasibility (observed: transport 100x100 → INFEASIBLE when every row
  // was perturbed). Slack rows — including explicit upper-bound rows that
  // create the degenerate 0-rhs cases in 0-1 MILP nodes — are where ratio
  // ties need breaking.
  //
  //   delta_i = (1e-10 + 1e-12*|b_i|) * (1 + i%1021)
  //
  // Absolute floor: O(1) knapsack nodes. Relative term: above double ULP when
  // |b_i| is O(1e7). Index mod 1021 caps max absolute delta near 1e-7, under
  // Phase I's feasibility_tol*10. Undone after Phase II via xB = B^{-1} b_true.
  StandardLp work = lp;
  {
    const bool log_lex = std::getenv("SOVEREIGN_DBG_CYCLE") != nullptr;
    int n_perturbed = 0;
    for (int i = 0; i < work.m; ++i) {
      const int logical_col = work.logical_basis[static_cast<std::size_t>(i)];
      const bool has_slack =
          logical_col >= 0 &&
          !is_artificial(work.names[static_cast<std::size_t>(logical_col)]);
      if (!has_slack) continue;
      const double bi = work.b[static_cast<std::size_t>(i)];
      const double idx = static_cast<double>((i % 1021) + 1);
      const double delta = (1e-10 + 1e-12 * std::abs(bi)) * idx;
      work.b[static_cast<std::size_t>(i)] = bi + delta;
      ++n_perturbed;
      if (log_lex && n_perturbed <= 8) {
        std::cerr << "[lex] row=" << i << " |b|=" << std::abs(bi)
                  << " delta=" << delta
                  << " changed=" << (work.b[static_cast<std::size_t>(i)] != bi)
                  << "\n";
      }
    }
    if (log_lex) {
      std::cerr << "[lex] perturbed_slack_rows=" << n_perturbed
                << " of m=" << work.m << "\n";
    }
  }

  SimplexState st;
  st.lp = &work;
  st.tol.feasibility = opt.feasibility_tol;
  st.tol.optimality = opt.optimality_tol;
  st.tol.pivot = opt.pivot_tol;
  st.refactor_every = opt.refactor_every;

  if (!initial_basis(st)) {
    result.status = SolverStatus::Error;
    result.message = "Failed to construct an initial logical basis.";
    return result;
  }

  // Phase I objective
  std::vector<double> c1(static_cast<std::size_t>(lp.n), 0.0);
  bool has_art = false;
  for (int j = 0; j < lp.n; ++j) {
    if (is_artificial(lp.names[static_cast<std::size_t>(j)])) {
      c1[static_cast<std::size_t>(j)] = 1.0;
      has_art = true;
    }
  }

  std::string detail;
  if (has_art) {
    const PhaseStatus ps =
        run_phase(st, c1, opt.max_iterations, true, &detail);
    result.iterations = st.iterations;
    if (ps == PhaseStatus::IterationLimit) {
      result.status = SolverStatus::IterationLimit;
      result.message = "Phase I hit the iteration limit (" +
                       std::to_string(opt.max_iterations) +
                       " iterations) without finding a feasible basis. " + detail;
      return result;
    }
    if (ps == PhaseStatus::Singular) {
      result.status = SolverStatus::NumericalError;
      result.message = "Phase I basis became numerically singular. " + detail;
      return result;
    }
    if (ps == PhaseStatus::Unbounded) {
      // Phase I should not be unbounded for artificial min >= 0
      result.status = SolverStatus::NumericalError;
      result.message =
          "Phase I reported unbounded, which is impossible for a minimize-artificial "
          "Phase I objective. The basis is numerically corrupt, not the model.";
      return result;
    }
    double phase1_obj = 0.0;
    for (int i = 0; i < lp.m; ++i) {
      const int col = st.basis[static_cast<std::size_t>(i)];
      if (is_artificial(lp.names[static_cast<std::size_t>(col)])) {
        phase1_obj += st.xB[static_cast<std::size_t>(i)];
      }
    }
    if (phase1_obj > opt.feasibility_tol * 10.0) {
      result.status = SolverStatus::Infeasible;
      result.message = "LP is infeasible (Phase I objective > 0).";
      result.iterations = st.iterations;
      return result;
    }

    // Remove artificials from basis if possible (drive to nonbasic).
    for (int i = 0; i < lp.m; ++i) {
      const int col = st.basis[static_cast<std::size_t>(i)];
      if (!is_artificial(lp.names[static_cast<std::size_t>(col)])) continue;
      if (std::abs(st.xB[static_cast<std::size_t>(i)]) > opt.feasibility_tol) {
        result.status = SolverStatus::Infeasible;
        result.message = "Artificial variable remains positive after Phase I.";
        return result;
      }
      // Try pivot artificial out with any non-artificial nonbasic that has nonzero
      bool replaced = false;
      for (int j : st.nonbasic) {
        if (is_artificial(lp.names[static_cast<std::size_t>(j)])) continue;
        std::vector<double> d;
        if (!pivot_column(st, j, d)) continue;
        if (std::abs(d[static_cast<std::size_t>(i)]) > opt.pivot_tol) {
          if (perform_pivot(st, j, i, d)) {
            replaced = true;
            break;
          }
        }
      }
      if (!replaced) {
        // Keep zero artificial in basis (degenerate feasible).
        result.warnings.push_back(
            "Zero-valued artificial retained in basis (degenerate).");
      }
    }
  }

  // Phase II: force artificial costs huge / keep them out of pricing if basic at 0
  std::vector<double> c2 = lp.c;
  for (int j = 0; j < lp.n; ++j) {
    if (is_artificial(lp.names[static_cast<std::size_t>(j)])) {
      c2[static_cast<std::size_t>(j)] = 0.0;
    }
  }

  const PhaseStatus ps2 = run_phase(st, c2, opt.max_iterations, false, &detail);
  result.iterations = st.iterations;
  if (ps2 == PhaseStatus::Singular) {
    result.status = SolverStatus::NumericalError;
    result.message = "Phase II basis became numerically singular. " + detail;
    return result;
  }
  if (ps2 == PhaseStatus::IterationLimit) {
    result.status = SolverStatus::IterationLimit;
    result.message = "Phase II hit the iteration limit (" +
                     std::to_string(opt.max_iterations) +
                     " iterations) before reaching optimality. No proven solution.";
    return result;
  }
  if (ps2 == PhaseStatus::Unbounded) {
    result.status = SolverStatus::Unbounded;
    result.message = "LP is unbounded.";
    return result;
  }

  // Undo lex RHS perturbation: refactor the final basis and solve against
  // the true (unperturbed) right-hand side so the reported primal/objective
  // are exact for the original LP.
  if (!refactor_basis(st)) {
    result.status = SolverStatus::Error;
    result.message = "Failed to refactor final basis after lex perturbation.";
    return result;
  }
  st.xB = lp.b;
  if (!ftran(st, st.xB)) {
    result.status = SolverStatus::Error;
    result.message = "Failed to recover solution against unperturbed RHS.";
    return result;
  }

  // Recover primal for structural variables
  std::vector<double> x(static_cast<std::size_t>(lp.n), 0.0);
  for (int i = 0; i < lp.m; ++i) {
    x[static_cast<std::size_t>(st.basis[static_cast<std::size_t>(i)])] =
        clamp_nonnegative(st.xB[static_cast<std::size_t>(i)], opt.feasibility_tol);
  }

  // ---------------------------------------------------------------------
  // Optimality certificate.
  //
  // run_phase() returned Optimal because no nonbasic column had a reduced cost
  // below -optimality_tol, but that was measured on the *perturbed* right-hand
  // side and against the eta chain. We have since refactorized and re-solved
  // against the true b, so the basis is not literally the one that was proven.
  // Claiming OPTIMAL without re-checking means a basis that silently lost dual
  // feasibility still gets reported as a proof. Measure it.
  //
  // Primal:   ||b - Ax||_inf / (1 + ||b||_inf)
  // Dual:     worst reduced cost over nonbasic columns (must be >= -tol)
  // Gap:      |c'x - b'y| / (1 + |c'x|), with y from the true dual B^{-T} c_B
  // ---------------------------------------------------------------------
  double residual = 0.0;
  {
    std::vector<double> Ax;
    lp.A.multiply(x, Ax);
    double bnorm = 1.0;
    for (int i = 0; i < lp.m; ++i) bnorm = std::max(bnorm, std::abs(lp.b[static_cast<std::size_t>(i)]));
    for (int i = 0; i < lp.m; ++i) {
      residual = std::max(residual, std::abs(lp.b[static_cast<std::size_t>(i)] - Ax[static_cast<std::size_t>(i)]));
    }
    residual /= bnorm;
  }
  result.primal_residual = residual;

  std::vector<double> cB(static_cast<std::size_t>(lp.m), 0.0);
  for (int i = 0; i < lp.m; ++i) {
    cB[static_cast<std::size_t>(i)] = lp.c[static_cast<std::size_t>(st.basis[static_cast<std::size_t>(i)])];
  }
  std::vector<double> y = cB;
  if (!btran(st, y)) {
    result.status = SolverStatus::NumericalError;
    result.message =
        "Could not recompute the dual vector from the final basis; optimality cannot be "
        "certified. The basis factorization is numerically unreliable.";
    return result;
  }

  double worst_rc = 0.0;
  for (int j : st.nonbasic) {
    if (is_artificial(lp.names[static_cast<std::size_t>(j)])) continue;
    double aj_pi = 0.0;
    for (int p = lp.A.col_ptr[static_cast<std::size_t>(j)];
         p < lp.A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
      aj_pi += lp.A.values[static_cast<std::size_t>(p)] *
               y[static_cast<std::size_t>(lp.A.row_idx[static_cast<std::size_t>(p)])];
    }
    worst_rc = std::min(worst_rc, lp.c[static_cast<std::size_t>(j)] - aj_pi);
  }
  result.dual_residual = std::max(0.0, -worst_rc);

  double cxs = 0.0;
  for (int j = 0; j < lp.n; ++j) {
    cxs += lp.c[static_cast<std::size_t>(j)] * x[static_cast<std::size_t>(j)];
  }
  double bys = 0.0;
  for (int i = 0; i < lp.m; ++i) {
    bys += lp.b[static_cast<std::size_t>(i)] * y[static_cast<std::size_t>(i)];
  }
  result.duality_gap = std::abs(cxs - bys) / (1.0 + std::abs(cxs));

  const bool primal_ok = residual <= opt.feasibility_tol;
  const bool dual_ok = -worst_rc <= opt.optimality_tol;
  const bool gap_ok = result.duality_gap <= std::max(opt.optimality_tol, 1e-9);

  if (!primal_ok || !dual_ok || !gap_ok) {
    std::ostringstream oss;
    oss << "Revised simplex finished iterating but the final basis does not certify "
        << "optimality. Relative primal residual = " << residual
        << " (tol " << opt.feasibility_tol << "); most negative reduced cost = " << worst_rc
        << " (tol -" << opt.optimality_tol << "); relative duality gap = "
        << result.duality_gap << " (tol " << opt.optimality_tol << "). ";
    if (!primal_ok) {
      oss << "Primal infeasible, so the point is not even usable. ";
    } else if (!dual_ok) {
      oss << "Primal feasible but the basis is dual infeasible, so this is a vertex "
             "that may be improvable; optimality is NOT proven. ";
    } else {
      oss << "Primal and dual feasible but the duality gap is still open. ";
    }
    oss << "Downgrading to NUMERICAL_ERROR rather than reporting OPTIMAL.";

    result.status = SolverStatus::NumericalError;
    result.message = oss.str();
      oss << "Primal and dual feasible but the duality gap is still open. ";
      oss << "Downgrading to NUMERICAL_ERROR rather than reporting OPTIMAL.";
      result.status = SolverStatus::NumericalError;
    }

    result.message = oss.str();
    // Still publish the primal so a caller can inspect it, but flag that the
    // objective is unverified. has_objective_value stays false so nothing
    // downstream can treat the number as an answer.
    for (int j = 0; j < lp.n_structural; ++j) {
      const double y_scaled = x[static_cast<std::size_t>(j)];
      const double yv = lp.col_scale[static_cast<std::size_t>(j)] * y_scaled;
      result.primal[lp.names[static_cast<std::size_t>(j)]] =
          lp.shift[static_cast<std::size_t>(j)] + yv;
    }
    return result;
  }

  double obj = original.objective.constant;
  for (int j = 0; j < lp.n_structural; ++j) {
    const double y_scaled = x[static_cast<std::size_t>(j)];
    const double y = lp.col_scale[static_cast<std::size_t>(j)] * y_scaled;
    const double xv = lp.shift[static_cast<std::size_t>(j)] + y;
    result.primal[lp.names[static_cast<std::size_t>(j)]] = xv;
    auto it = original.objective.linear.find(lp.names[static_cast<std::size_t>(j)]);
    if (it != original.objective.linear.end()) obj += it->second * xv;
  }

  result.status = SolverStatus::Optimal;
  result.has_objective_value = true;
  result.objective_value = obj;
  result.message = "Optimal solution found by revised simplex (product-form basis updates).";
  return result;
}

}  // namespace

RevisedSimplexSolver::RevisedSimplexSolver(RevisedSimplexOptions options)
    : options_(std::move(options)) {}

SolverResult RevisedSimplexSolver::solve(const OptimizationModel& model) const {
  SolverResult result;
  try {
    if (model.problem_type != ProblemType::LP) {
      result.status = SolverStatus::Error;
      result.message = "RevisedSimplexSolver expects problem_type=LP.";
      return result;
    }
    if (!model.objective.quadratic.empty()) {
      result.status = SolverStatus::Error;
      result.message = "LP model must not contain quadratic terms.";
      return result;
    }
    for (const auto& v : model.variables) {
      if (v.type != VariableType::Continuous) {
        result.status = SolverStatus::Error;
        result.message = "LP variables must be continuous (got non-continuous).";
        return result;
      }
    }

    StandardLp lp = build_standard_form(model, options_.enable_scaling);
    return solve_standard(lp, options_, model);
  } catch (const std::exception& ex) {
    result.status = SolverStatus::Error;
    result.message = ex.what();
    return result;
  }
}

}  // namespace sovereign
