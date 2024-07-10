#include "app.hpp"

#include <trace-utils/logger.hpp>

namespace trace_utils::app {
App::App(const std::string& name, const std::string& description): name_{name}, description_{description} {
    log()->debug("Initializing app with name={}, description={}", name_, description_);
}
    
void NamespaceApp::setup_args(CLI::App* app) {
    parser = create_subcommand(app);
    // parser->set_help_flag("--help", "Expand help");
    // parser->set_help_all_flag("--help-all", "Expand all help");
}

void NamespaceApp::run(CLI::App* app) {
    for (auto& sub : apps) {
        if (parser->got_subcommand(sub->name())) { 
            sub->setup();
            sub->call(parser);
        }
    }
}    
} // namespace trace_utils::app
