/*-------------------------------------------------------------------------------
  This file is part of generalized random forest (grf).

  grf is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  grf is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with grf. If not, see <http://www.gnu.org/licenses/>.
 #-------------------------------------------------------------------------------*/

#ifndef GRF_BALANCEDREGRESSIONSPLITTINGRULE_H
#define GRF_BALANCEDREGRESSIONSPLITTINGRULE_H

#include "commons/Data.h"
#include "splitting/SplittingRule.h"
#include "tree/Tree.h"

namespace grf {

class BalancedRegressionSplittingRule final: public SplittingRule {
public:
  BalancedRegressionSplittingRule(size_t max_num_unique_values,
                          double alpha,
                          double imbalance_penalty);

  ~RegressionSplittingRule();

  bool find_best_split(const Data& data,
                       size_t node,
                       const std::vector<size_t>& possible_split_vars,
                       const Eigen::ArrayXXd& responses_by_sample,
                       const std::vector<std::vector<size_t>>& samples,
                       std::vector<size_t>& split_vars,
                       std::vector<double>& split_values,
                       std::vector<bool>& send_missing_left);

private:
  void find_best_split_value(const Data& data,
                             size_t node,
                             size_t var,
                             double weight_sum_node,
                             double sum_node,
                             size_t size_node,
                             size_t min_child_size,
                             double& best_value,
                             size_t& best_var,
                             double& best_decrease,
                             bool& best_send_missing_left,
                             const Eigen::ArrayXXd& responses_by_sample,
                             const std::vector<std::vector<size_t>>& samples,
                             arma::vec &target_weight_sum,
                             arma::mat &target_weight_left_sum,
                             const std::string &target_weight_penalty_metric,
                             const double &target_weight_penalty_rate);

  size_t* counter;
  double* sums;
  double* weight_sums;

  double alpha;
  double imbalance_penalty;
  double *target_left_weights;

  DISALLOW_COPY_AND_ASSIGN(BalancedRegressionSplittingRule);
};

} // namespace grf

#endif //GRF_BALANCEDREGRESSIONSPLITTINGRULE_H
