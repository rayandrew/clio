#include <trace-utils/trace/tencent.hpp>
#include <trace-utils/trace/alibaba.hpp>

#include <fmt/core.h>

#include <CLI/CLI.hpp>

#include <trace-utils/logger.hpp>
#include <cpptrace/cpptrace.hpp>

#include "./tencent/tencent.hpp"
#include "./alibaba/alibaba.hpp"
#include "./stats/stats.hpp"

#ifdef __unix__
#include <sys/wait.h>
#include <cstring>
#include <unistd.h>

#include <csignal>
#include <indicators/cursor_control.hpp>

namespace
{
    volatile std::sig_atomic_t shutdown;
}

std::sig_atomic_t signaled = 0;

// This is just a utility I like, it makes the pipe API more expressive.
struct pipe_t
{
    union
    {
        struct
        {
            int read_end;
            int write_end;
        };
        int data[2];
    };
};

void do_signal_safe_trace(cpptrace::frame_ptr *buffer, std::size_t count)
{
    // Setup pipe and spawn child
    pipe_t input_pipe;
    pipe(input_pipe.data);
    const pid_t pid = fork();
    if (pid == -1)
    {
        return; /* Some error occurred */
    }
    if (pid == 0)
    { // child
        dup2(input_pipe.read_end, STDIN_FILENO);
        close(input_pipe.read_end);
        close(input_pipe.write_end);
        execl("trace-utils-stack-trace", "trace-utils-stack-trace", nullptr);
        const char *exec_failure_message = "exec(trace-utils-stack-trace) failed: Make sure the signal_tracer executable is in "
                                           "the current working directory and the binary's permissions are correct.\n";
        write(STDERR_FILENO, exec_failure_message, strlen(exec_failure_message));
        indicators::show_console_cursor(true);
        std::exit(1);
    }
    // Resolve to safe_object_frames and write those to the pipe
    for (std::size_t i = 0; i < count; i++)
    {
        cpptrace::safe_object_frame frame;
        cpptrace::get_safe_object_frame(buffer[i], &frame);
        write(input_pipe.write_end, &frame, sizeof(frame));
    }
    close(input_pipe.read_end);
    close(input_pipe.write_end);
    // Wait for child
    waitpid(pid, nullptr, 0);
}

void handler(int signo, [[maybe_unused]] siginfo_t *info, [[maybe_unused]] void *context)
{
    // Print basic message
    const char *message = "SIGSEGV occurred:\n";
    write(STDERR_FILENO, message, strlen(message));
    // Generate trace
    constexpr std::size_t N = 100;
    cpptrace::frame_ptr buffer[N];
    std::size_t count = cpptrace::safe_generate_raw_trace(buffer, N);
    do_signal_safe_trace(buffer, count);
    // Up to you if you want to exit or continue or whatever
    shutdown = signo;
    indicators::show_console_cursor(true);
    std::exit(1);
}

void warmup_cpptrace()
{
    // This is done for any dynamic-loading shenanigans
    cpptrace::frame_ptr buffer[10];
    [[maybe_unused]] std::size_t count = cpptrace::safe_generate_raw_trace(buffer, 10);
    cpptrace::safe_object_frame frame;
    cpptrace::get_safe_object_frame(buffer[0], &frame);
}
#endif

int main(int argc, char *argv[])
{
#ifdef __unix__
    warmup_cpptrace();
    // Setup signal handler
    struct sigaction action = {0};
    action.sa_flags = 0;
    action.sa_sigaction = &handler;
    if (sigaction(SIGSEGV, &action, NULL) == -1)
    {
        perror("sigaction");
    }
#endif
    cpptrace::register_terminate_handler();

    trace_utils::logger::create();

    CLI::App app;
    app.set_help_flag("-h,--help", "Expand help");
    app.set_help_all_flag("--help-all", "Expand all help");

    try
    {
        trace_utils::app::TencentApp tencent_app;
        tencent_app.setup_args(&app);

        trace_utils::app::StatsApp stats_app;
        stats_app.setup_args(&app);

        trace_utils::app::AlibabaApp alibaba_app;
        alibaba_app.setup_args(&app);

        app.parse(argc, argv);

        tencent_app(&app);
        stats_app(&app);
        alibaba_app(&app);
    }
    catch (const CLI::CallForHelp &err)
    {
        trace_utils::log()->info("Help\n\n{}", app.help(""));
        return 1;
    }
    catch (const CLI::CallForAllHelp &err)
    {
        trace_utils::log()->info("Help All\n\n{}", app.help("", CLI::AppFormatMode::All));
        return 1;
    }
    catch (const CLI::Error &err)
    {
        trace_utils::log()->error("CLI::Error: {}", err.what());
        trace_utils::log()->error("Run --help/--help-all to see options");
        return 1;
    }
    catch (const trace_utils::Exception &err)
    {
        trace_utils::log()->error("Trace Utils Error:\n\n  {}", err.what());

        err.trace().print(std::cerr, cpptrace::isatty(cpptrace::stderr_fileno));
        return 1;
    }
    // Commented out, error message not informative
    // catch (const std::exception &err)
    // {
    //     trace_utils::log()->error("Error: {}", err.what());
    //     err.trace().print(std::cerr, cpptrace::isatty(cpptrace::stderr_fileno));
    //     return 1;
    // }

    return 0;
}
