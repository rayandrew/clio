#ifndef __TRACE_UTILS_APP_TECTONIC_HPP__
#define __TRACE_UTILS_APP_TECTONIC_HPP__

#include "../app.hpp"

namespace trace_utils::app {
class TectonicApp : public NamespaceApp {
 public:
  TectonicApp();

  virtual void setup_args(CLI::App* app) override;
};
}  // namespace trace_utils::app

#endif
