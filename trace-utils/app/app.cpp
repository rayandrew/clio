#include "app.hpp"

#include <trace-utils/logger.hpp>

namespace trace_utils::app {
App::App(const std::string& name, const std::string& description): name_{name}, description_{description} {
    log()->debug("Initializing app with name={}, description={}", name_, description_);
}
    
void NamespaceApp::setup(CLI::App* app) {
    parser = create_subcommand(app);
}

void NamespaceApp::run(CLI::App* app) {
    for (auto& sub : apps) {
        if (parser->get_subcommand(sub->parser)) {
            sub->call(parser);
        }
    }
}    
} // namespace trace_utils::app
