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

#ifndef GRF_BALANCEDPROBABILITYSPLITTINGRULEFACTORY_H
#define GRF_BALANCEDPROBABILITYSPLITTINGRULEFACTORY_H

#include <vector>

#include "commons/globals.h"
#include "splitting/factory/SplittingRuleFactory.h"

namespace grf {

/**
 * A factory that produces balanced classification splitting rules.
 *
 * In addition to performing standard regression splits, this rule applies
 * a penalty to avoid splits too imbalanced to the protected features.
 */
class BalancedProbabilitySplittingRuleFactory final: public SplittingRuleFactory {
public:
  BalancedProbabilitySplittingRuleFactory(size_t num_classes);
  std::unique_ptr<SplittingRule> create(size_t max_num_unique_values,
                                        const TreeOptions& options) const;

private:
  size_t num_classes;

  DISALLOW_COPY_AND_ASSIGN(BalancedProbabilitySplittingRuleFactory);
};

} // namespace grf

#endif //GRF_BALANCEDPROBABILITYSPLITTINGRULEFACTORY_H
