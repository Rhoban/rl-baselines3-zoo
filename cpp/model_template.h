#pragma once

#include "baselines3_models/predictor.h"
#include "torch/script.h"

namespace baselines3_models {
class CLASS_NAME : public Predictor {
public:
  CLASS_NAME();

  /**
   * Observation space is:
   * OBSERVATION_SPACE
   */
  torch::Tensor preprocess_observation(torch::Tensor &observation) override;

  /**
   * Action space is:
   * ACTION_SPACE
   */
  torch::Tensor process_action(torch::Tensor &action) override;
};
} // namespace baselines3_models