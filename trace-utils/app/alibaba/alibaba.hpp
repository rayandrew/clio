#ifndef __TRACE_UTILS_APP_ALIBABA_HPP__
#define __TRACE_UTILS_APP_ALIBABA_HPP__

#include "../app.hpp"

namespace trace_utils::app
{
    class AlibabaApp : public NamespaceApp
    {
    public:
        AlibabaApp();

        virtual void setup_args(CLI::App *app) override;
    };
} // namespace trace_utils::app

#endif
