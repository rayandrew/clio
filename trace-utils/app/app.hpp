#ifndef __TRACE_UTILS_APP_APP_HPP__
#define __TRACE_UTILS_APP_APP_HPP__

#include <initializer_list>
#include <type_traits>
#include <memory>

#include <CLI/CLI.hpp>


namespace trace_utils::app {
class NamespaceApp;
    
class App {
public:
    App(const std::string& name, const std::string& description = "");

    virtual ~App() = default;

    virtual void setup_args(CLI::App* app) = 0;

    inline virtual void setup() { }
    
    inline std::string name() { return name_; }
    inline std::string description() { return description_; }
    
    virtual void run(CLI::App* app) = 0;
    
    inline void call(CLI::App* app) {
        if (app->got_subcommand(parser)) run(app);
    }
    
    inline void operator()(CLI::App* app) {
        call(app);
    }

protected:
    inline virtual CLI::App* create_subcommand(CLI::App* app) {
        return app->add_subcommand(name(), description());
    }

    friend class NamespaceApp;

protected:
    CLI::App* parser;

    std::string name_;
    std::string description_;
};

class NamespaceApp : public App {
public:
    using App::App;
    
    virtual void setup_args(CLI::App* app) override;
    virtual void run(CLI::App* app) override;

protected:
    template<typename T, typename... Args, std::enable_if_t<std::is_base_of_v<App, T>, bool> = true>
    void add(Args&&... args) {
        apps.emplace_back(std::make_unique<T>(std::forward<Args>(args)...));
    }

protected:
    std::vector<std::unique_ptr<App>> apps;
};
} // namespace trace_utils::app

#endif
