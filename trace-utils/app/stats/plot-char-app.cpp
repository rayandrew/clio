#include "plot-char-app.hpp"

#include <string_view>
#include <charconv>
#include <iostream>

#include <glob/glob.h>
#include <natural_sort.hpp>
#include <oneapi/tbb.h>

#include <fmt/std.h>
#include <fmt/ranges.h>
#include <fmt/chrono.h>

#include <csv2/writer.hpp>
#include <csv2/reader.hpp>

#include <mp-units/systems/si/si.h>
#include <mp-units/systems/isq/isq.h>

#include <indicators/block_progress_bar.hpp>
#include <indicators/cursor_control.hpp>
#include <indicators/termcolor.hpp>

#include <trace-utils/logger.hpp>
#include <trace-utils/characteristic.hpp>
#include <trace-utils/trace.hpp>
#include <trace-utils/trace/replayer.hpp>
#include <trace-utils/utils.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
#include <tbb/concurrent_vector.h>
#include <tbb/blocked_range.h>

const std::vector<std::string> NORMALIZED_METRICS = {
    "iops", "read_iops", "write_iops",
    "read_ratio", "write_ratio",
    "iat_avg", "read_iat_avg", "write_iat_avg",
    "size_avg", "read_size_avg", "write_size_avg"};

namespace trace_utils::app::stats::calculate
{
    namespace plot_char
    {
        const char *name = "plot-char";
        const char *description = "Plot Characteristic";
    } // namespace plot_char

    PlotCharApp::PlotCharApp() : App(plot_char::name, plot_char::description)
    {
    }

    PlotCharApp::~PlotCharApp()
    {
        indicators::show_console_cursor(true);
    }

    void PlotCharApp::setup_args(CLI::App *app)
    {
        parser = create_subcommand(app);
        parser->add_option("-i,--input", input, "Input File")->required();
        parser->add_option("-o,--output", output, "Output directory")->required();
    }

    void PlotCharApp::setup()
    {
        output = fs::weakly_canonical(output);
        fs::create_directories(output);
    }

    const std::vector<std::string> NORMALIZED_METRICS = {
        "iops", "read_iops", "write_iops",
        "read_ratio", "write_ratio",
        "iat_avg", "read_iat_avg", "write_iat_avg",
        "size_avg", "read_size_avg", "write_size_avg"};

    void min_max_scale(const std::vector<double> &values, std::vector<double> &normalized_values)
    {
        auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
        double min = *min_it;
        double max = *max_it;
        for (const auto &value : values)
        {
            normalized_values.push_back((value - min) / (max - min));
        }
    }

    void generate_cdf(const std::vector<double> &data, const fs::path &output_path)
    {
        // Sort the data
        std::vector<double> sorted_data = data;
        std::sort(sorted_data.begin(), sorted_data.end());

        // Compute the CDF
        std::vector<double> cdf_values(sorted_data.size());
        std::iota(cdf_values.begin(), cdf_values.end(), 1); // Fill with 1, 2, ..., n
        std::transform(cdf_values.begin(), cdf_values.end(), cdf_values.begin(),
                       [n = sorted_data.size()](double x)
                       { return x / n; });

        fs::create_directories(output_path.parent_path());

        std::ofstream cdf_file(output_path);
        for (size_t i = 0; i < sorted_data.size(); ++i)
        {
            cdf_file << sorted_data[i] << "\t" << cdf_values[i] << "\n";
        }
    }

    void process_csv(const fs::path &input, const fs::path &output)
    {
        csv2::Reader<csv2::delimiter<','>, csv2::quote_character<'"'>, csv2::first_row_is_header<true>> csv;
        if (csv.mmap(input.string()))
        {
            std::unordered_map<size_t, size_t> column_mapping;
            std::vector<std::string> column_names;
            std::vector<std::vector<double>> columns;

            // Read header separately
            const auto header = csv.header();
            size_t col_idx = 0;
            for (const auto &cell : header)
            {
                std::string column_name;
                cell.read_value(column_name);
                auto it = std::find(NORMALIZED_METRICS.begin(), NORMALIZED_METRICS.end(), column_name);
                if (it != NORMALIZED_METRICS.end())
                {
                    column_mapping[col_idx] = columns.size();
                    column_names.push_back(column_name);
                    columns.emplace_back();
                }
                ++col_idx;
            }

            // Process rows
            for (const auto &row : csv)
            {
                col_idx = 0;
                for (const auto &cell : row)
                {
                    if (column_mapping.find(col_idx) != column_mapping.end())
                    {
                        std::string cell_value{""};
                        cell.read_value(cell_value);
                        double value = std::stod(cell_value);
                        columns[column_mapping[col_idx]].push_back(value);
                    }
                    ++col_idx;
                }
            }

            for (size_t col_idx = 0; col_idx < columns.size(); ++col_idx)
            {
                const auto &column_name = column_names[col_idx];
                const auto &column_data = columns[col_idx];

                double min_value = *std::min_element(column_data.begin(), column_data.end());
                double max_value = *std::max_element(column_data.begin(), column_data.end());

                std::vector<double> normalized_data;
                std::transform(column_data.begin(), column_data.end(), std::back_inserter(normalized_data),
                               [min_value, max_value](double x)
                               { return (x - min_value) / (max_value - min_value); });
                // Create directories for raw and normalized data under both "line" and "pdf"
                fs::create_directories(output / "line" / column_name / "raw");
                fs::create_directories(output / "line" / column_name / "normalized");
                fs::create_directories(output / "cdf" / column_name / "raw");
                fs::create_directories(output / "cdf" / column_name / "normalized");

                // This will write .dat file so GNUPLOT can process and plot as line
                std::ofstream raw_file((output / "line" / column_name / "raw" / "data.dat").string());
                std::ofstream normalized_file((output / "line" / column_name / "normalized" / "data.dat").string());

                for (size_t i = 0; i < column_data.size(); ++i)
                {
                    raw_file << i << "\t" << column_data[i] << "\n";
                    normalized_file << i << "\t" << normalized_data[i] << "\n";
                }

                generate_cdf(column_data, output / "cdf" / column_name / "raw" / "data.dat");
                generate_cdf(normalized_data, output / "cdf" / column_name / "normalized" / "data.dat");
            }
        }
    }

    void PlotCharApp::run([[maybe_unused]] CLI::App *app)
    {
        std::cout << "Hello from PlotCharApp" << std::endl;

        auto input_path = fs::canonical(input);

        // if input_path does not exist, or is not csv, return
        if (!fs::exists(input_path) || input_path.extension() != ".csv")
        {
            std::cerr << "Error, Expecting to get a csv file! " << input_path << std::endl;
            return;
        }
        // delay file parallelism for later, call this per char.csv file

        // std::vector<fs::path> paths;

        // for (const auto &entry : fs::recursive_directory_iterator(input_path))
        // {
        //     if (entry.is_regular_file() && entry.path().extension() == ".csv")
        //     {
        //         paths.push_back(entry.path());
        //     }
        // }
        // std::cout << "Found " << paths.size() << " csv files" << std::endl;

        // // if > 1 file, return error
        // if (paths.size() != 1)
        // {
        //     std::cerr << "Error, Expecting to get 1 csv file only! " << input_path << std::endl;
        //     return;
        // }

        process_csv(input_path, output);

        // tbb::parallel_for(tbb::blocked_range<size_t>(0, paths.size()), [&](const tbb::blocked_range<size_t> &r)
        //                   {
        // for (size_t i = r.begin(); i != r.end(); ++i) {
        //     process_csv(paths[i], output);
        // } });
    }
} // namespace trace_utils::app::stats::calculate
