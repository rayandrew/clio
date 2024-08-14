#include "trace-finder-v1.hpp"

#include <indicators/block_progress_bar.hpp>
#include <indicators/cursor_control.hpp>
#include <indicators/termcolor.hpp>

#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <numeric>

namespace trace_utils::app::trace_finder {
namespace trace_finder_v1 {
const char *name = "v1";
const char *description = "Trace Finder v1";

// float STABILITY_THRESHOLD = 14.0;
// float DRIFT_THRESHOLD = 50.0;
// float GROUP_THRESHOLD = 250.0;
// float GROUP_OFFSET = 50.0;
// int ROLLING_WINDOW = 10;

struct Row {
    int index;
    float value;
    int stability;
    int stability_group;
    int group;
    std::map<int, int> mode;
    std::map<int, int> next_mode;
    int prev_group;
};

int determine_group(float value, float threshold, float offset) {
    if (value < 0)
    {
        return 0;
    };

    if (value <= offset)
    {
        return 1;
    };

    return int((value - offset) / threshold) + 2;
}

std::map<int, int> calc_mode_row(const std::vector<Row> &rows) {
    std::map<int, int> mode_result;
    for (const auto &row : rows)
    {
        if (mode_result.find(row.group) == mode_result.end())
        {
            mode_result[row.group] = 0;
        }
        mode_result[row.group] += 1;
    }
    return mode_result;
}

std::map<int, int> get_similar_modes(const std::map<int, int> &mode, int criteria, int diff, bool include_self = true) {
    std::vector<int> diffs;
    if (include_self)
    {
        diffs.push_back(0);
    }

    if (diff > 0)
    {
        for (int i = 1; i <= diff; ++i)
        {
            diffs.push_back(i);
            diffs.push_back(-i);
        }
    }
    std::map<int, int> result;
    for (int diff : diffs)
    {
        result[criteria + diff] = mode.find(criteria + diff) != mode.end() ? mode.at(criteria + diff) : 0;
    }
    return result;
}

std::vector<Row> read_csv(const std::string &characteristic_file, const std::string &metric) {
    std::ifstream file(characteristic_file);
    std::string line;
    std::vector<Row> rows;
    std::map<std::string, int> header_map;
    bool header = true;

    if (!file)
    {
        std::cerr << "File not found: " << characteristic_file << std::endl;
        exit(1);
    }

    int index = -1;
    while (std::getline(file, line))
    {
        std::istringstream ss(line);
        std::string value;
        Row row;
        row.index = index++;

        if (header)
        {
            int col_index = 0;
            while (std::getline(ss, value, ','))
            {
                header_map[value] = col_index++;
            }
            header = false;

            if (header_map.find(metric) == header_map.end())
            {
                std::cerr << "Metric not found in header: " << metric << std::endl;
                exit(1);
            }
            continue;
        }

        int col_index = 0;
        while (std::getline(ss, value, ','))
        {
            if (col_index == header_map[metric])
            {
                row.value = std::stod(value);
            }
            col_index++;
        }

        rows.push_back(row);
    }

    return rows;
}

int lookback_nearest_drift_idx(const std::vector<Row> &rows, int drift_idx, const std::string &key, int criteria) {
    int new_drift_idx = drift_idx;
    for (int i = new_drift_idx; i >= 0; --i)
    {
        int value;
        if (key == "group")
        {
            value = rows[i].group;
        }
        else
        {
            exit(1);
        }
        if (value == criteria)
        {
            new_drift_idx = i;
            break;
        }
    }
    return new_drift_idx;
}

int lookback_consecutive_drift_idx(const std::vector<Row> &rows, int drift_idx, const std::string &key, int criteria) {
    int new_drift_idx = drift_idx;
    for (int i = new_drift_idx; i >= 0; --i)
    {
        int prev_i = i - 1;

        int attr, prev_attr;
        if (key == "group")
        {
            attr = rows[i].group;
            prev_attr = rows[prev_i].group;
        }
        else
        {
            std::cout << "Key not yet implemented in lookback " << key << std::endl;
            exit(1);
        }

        if (attr != criteria && prev_attr == criteria)
        {
            continue;
        }
        if (attr == criteria && prev_attr != criteria)
        {
            new_drift_idx = i;
            break;
        }
    }
    return new_drift_idx;
}

int lookahead_nearest_drift_idx(const std::vector<Row> &rows, int drift_idx, const std::string &key, int criteria) {
    int new_drift_idx = drift_idx;
    for (std::size_t i = new_drift_idx + 1; i < rows.size(); ++i)
    {
        int value;
        if (key == "group")
        {
            value = rows[i].group;
        }
        else
        {
            exit(1);
        }
        if (value == criteria)
        {
            new_drift_idx = i;
            break;
        }
    }
    return new_drift_idx;
}

int lookahead_consecutive_drift_idx(const std::vector<Row> &rows, int drift_idx, const std::string &key, int criteria) {
    int new_drift_idx = drift_idx;

    for (std::size_t i = new_drift_idx; i < rows.size(); ++i)
    {
        int next_i = i + 1;

        int attr, next_attr;
        if (key == "group")
        {
            attr = rows[i].group;
            next_attr = rows[next_i].group;
        }
        else
        {
            std::cout << "Key not yet implemented in lookback " << key << std::endl;
            exit(1);
        }

        if (attr != criteria && next_attr == criteria)
        {
            continue;
        }

        if (attr == criteria && next_attr != criteria)
        {
            new_drift_idx = i;
            break;
        }
    }
    return new_drift_idx;
}

std::vector<std::pair<int, int>> collapse_intervals(const std::vector<std::pair<int, int>> &intervals, int threshold)
{
    if (intervals.empty())
    {
        return {};
    }

    // Sort intervals by the start value
    std::vector<std::pair<int, int>> sorted_intervals = intervals;
    std::sort(sorted_intervals.begin(), sorted_intervals.end());

    std::vector<std::pair<int, int>> collapsed;
    collapsed.push_back(sorted_intervals[0]);

    for (size_t i = 1; i < sorted_intervals.size(); ++i)
    {
        int start = sorted_intervals[i].first;
        int end = sorted_intervals[i].second;
        int last_end = collapsed.back().second;

        if (start - last_end <= threshold)
        {
            // Merge intervals
            collapsed.back().second = end;
        }
        else
        {
            // Add new interval to the result
            collapsed.push_back({start, end});
        }
    }

    return collapsed;
}

int get_highest_mode(std::map<int, int> mode)
{
    if (mode.empty())
    {
        return -1;
    }

    auto max_elem = std::max_element(
        mode.begin(), mode.end(),
        [](const std::pair<int, int> &a, const std::pair<int, int> &b)
        {
            return a.second < b.second;
        });

    return max_elem->first;
}

void process(const std::vector<Row> &rows, const fs::path &output_path, float stability_threshold, float rolling_window) {
    std::vector<int> potential_end_concepts;

    // Find potential stable concepts
    for (size_t i = 0; i < rows.size() - 1; ++i)
    {
        const auto &c = rows[i];
        const auto &n = rows[i + 1];
        if (c.stability >= stability_threshold && n.stability == 0)
        {
            // assert(c.index == i);
            potential_end_concepts.push_back(c.index);
        }
    }

    // Concept finder
    std::vector<std::pair<int, int>> possible_drifts;
    for (size_t i = 0; i < potential_end_concepts.size(); ++i)
    {
        int start = potential_end_concepts[i];
        for (size_t j = i + 1; j < potential_end_concepts.size(); ++j)
        {
            int end = potential_end_concepts[j];
            if (end - (start - rows[start].stability) < 3600)
            {
                possible_drifts.emplace_back(start, end);
            }
            else
            {
                break;
            }
        }
    }

    std::vector<std::tuple<int, int, std::string>> drifts;
    std::vector<std::pair<int, int>> recurring_drifts;
    std::map<int, std::map<int, int>> caches_modes;

    std::cout << "Possible drifts: " << possible_drifts.size() << std::endl;
    int counter = 0;
    int gradual_count = 0, sudden_count = 0, incremental_count = 0, recurring_count = 0;
    for (const auto &[start_idx, end_idx] : possible_drifts)
    {
        // if (counter % 1000 == 0)
        // {
        //     std::cout << "Processing drift " << counter << std::endl;
        // }
        counter += 1;

        int start_drift_idx = rows[start_idx].index;
        int start_mode = get_highest_mode(rows[start_drift_idx].mode);
        int start_from_stability = rows[start_drift_idx].index - rows[start_drift_idx].stability;
        if (rows[start_from_stability].group == start_mode)
        {
            start_drift_idx = start_from_stability;
        }
        else if (rows[start_drift_idx].group != start_mode)
        {
            start_drift_idx = lookback_nearest_drift_idx(rows, start_drift_idx, "group", start_mode);
            start_drift_idx = lookback_consecutive_drift_idx(rows, start_drift_idx, "group", start_mode);
        }
        const Row &start = rows[start_drift_idx];
        int end_drift_idx = end_idx;
        const Row &end = rows[end_drift_idx];

        start_mode = get_highest_mode(start.mode);
        int end_mode = get_highest_mode(end.mode);

        if (start_mode == end_mode)
        {
            int new_start_drift_idx = start_drift_idx;
            int new_end_drift_idx = end_drift_idx;
            if (rows[end_drift_idx].group != start_mode)
            {
                new_end_drift_idx = lookback_nearest_drift_idx(rows, end_drift_idx, "group", start_mode);
            }
            new_end_drift_idx = lookahead_consecutive_drift_idx(rows, new_end_drift_idx, "group", start_mode);

            // TODO: SLICE THIS NOT CREATE
            std::vector<Row> intermediate_rows(rows.begin() + start_drift_idx + 1, rows.begin() + end_drift_idx);
            int count_less_than_half = 0;
            for (const auto &c : intermediate_rows)
            {
                auto potential_c_modes = get_similar_modes(c.mode, start_mode, 1, true);
                int sum_potential_c_modes = 0;
                for (const auto &[_, value] : potential_c_modes)
                {
                    sum_potential_c_modes += value;
                }
                if (sum_potential_c_modes < (rolling_window / 2))
                {
                    ++count_less_than_half;
                }
            }

            if (count_less_than_half > 8)
            {
                drifts.emplace_back(new_start_drift_idx, new_end_drift_idx, "recurring");
                recurring_drifts.emplace_back(new_start_drift_idx, new_end_drift_idx);
                recurring_count += 1;
            }
        }
        else if (abs(start_mode - end_mode) > 1)
        {
            bool is_gradual = false;
            int end_current_drift_idx = start_idx;
            std::vector<Row> intermediate_rows(rows.begin() + end_current_drift_idx + 1, rows.begin() + end_drift_idx + 1);

            std::vector<int> potential_c_modes_l{0};
            std::vector<std::pair<int, int>> concepts;
            int start_gradual_idx = 0;

            for (size_t i = 0; i < intermediate_rows.size() - 1; ++i)
            {
                const auto &c = intermediate_rows[i];
                const auto &n = intermediate_rows[i + 1];
                auto potential_c_modes = caches_modes.find(c.index) != caches_modes.end() ? caches_modes[c.index] : get_similar_modes(c.mode, end_mode, 1, true);
                auto potential_n_modes = caches_modes.find(n.index) != caches_modes.end() ? caches_modes[n.index] : get_similar_modes(n.mode, end_mode, 1, true);
                caches_modes[c.index] = potential_c_modes;
                caches_modes[n.index] = potential_n_modes;
                int sum_potential_c_modes = 0;
                int sum_potential_n_modes = 0;
                for (const auto &[_, value] : potential_c_modes)
                {
                    sum_potential_c_modes += value;
                }
                for (const auto &[_, value] : potential_n_modes)
                {
                    sum_potential_n_modes += value;
                }
                potential_c_modes_l.push_back(sum_potential_c_modes);

                if (start_gradual_idx == 0 && sum_potential_c_modes >= (rolling_window / 2))
                {
                    start_gradual_idx = c.index;
                }

                if (sum_potential_c_modes >= (rolling_window / 2) && sum_potential_n_modes >= (rolling_window / 2))
                {
                    continue;
                }

                if (sum_potential_c_modes < (rolling_window / 2))
                {
                    if (start_gradual_idx != 0)
                    {
                        int end_gradual_idx = c.index;
                        concepts.emplace_back(start_gradual_idx, end_gradual_idx);
                        start_gradual_idx = 0;
                    }
                }
            }

            if (start_gradual_idx != 0)
            {
                concepts.emplace_back(start_gradual_idx, intermediate_rows.back().index);
            }

            auto cleaned_concepts = collapse_intervals(concepts, rolling_window);
            is_gradual = cleaned_concepts.size() > 1;

            if (is_gradual)
            {
                int new_start_drift_idx = lookback_consecutive_drift_idx(rows, start_drift_idx, "group", start_mode);
                int new_end_drift_idx;
                if (rows[end_drift_idx].group != end_mode)
                {
                    new_end_drift_idx = lookback_nearest_drift_idx(rows, end_drift_idx, "group", end_mode);
                }
                new_end_drift_idx = lookahead_consecutive_drift_idx(rows, end_drift_idx, "group", end_mode);
                gradual_count += 1;
                drifts.emplace_back(new_start_drift_idx, new_end_drift_idx, "gradual");
            }
            else
            {
                bool should_be_increasing = start_mode < end_mode;
                bool is_incremental = true;

                std::vector<int> ranges;
                for (int i = start_mode; i <= end_mode; ++i)
                {
                    ranges.push_back(i);
                }

                std::vector<std::map<int, int>> modes_l;
                for (const auto &c : intermediate_rows)
                {
                    std::map<int, int> modes;
                    for (const auto &mode : ranges)
                    {
                        modes[mode] = c.mode.count(mode) ? c.mode.at(mode) : 0;
                    }
                    modes_l.push_back(modes);
                }

                std::vector<double> y;
                std::vector<std::vector<double>> x;
                for (size_t i = 0; i < modes_l.size(); ++i)
                {
                    x.push_back({static_cast<double>(i), 1.0});
                    auto similar_m_start = get_similar_modes(modes_l[i], start_mode, 0, true);
                    double sum_similar_m_start = 0;
                    for (const auto &[_, value] : similar_m_start)
                    {
                        sum_similar_m_start += value;
                    }
                    y.push_back(sum_similar_m_start);
                }

                // Perform linear regression
                std::vector<std::vector<double>> xT(2, std::vector<double>(x.size()));
                for (size_t i = 0; i < x.size(); ++i)
                {
                    xT[0][i] = x[i][0];
                    xT[1][i] = x[i][1];
                }

                std::vector<std::vector<double>> xTx(2, std::vector<double>(2));
                xTx[0][0] = xT[0][0] * xT[0][0] + xT[0][1] * xT[0][1];
                xTx[0][1] = xT[0][0] * xT[1][0] + xT[0][1] * xT[1][1];
                xTx[1][0] = xTx[0][1];
                xTx[1][1] = xT[1][0] * xT[1][0] + xT[1][1] * xT[1][1];

                std::vector<double> xTy(2);
                xTy[0] = xT[0][0] * y[0] + xT[0][1] * y[1];
                xTy[1] = xT[1][0] * y[0] + xT[1][1] * y[1];

                std::vector<double> betas(2);
                betas[0] = (xTy[0] - xTx[0][1] * xTy[1]) / (xTx[0][0] - xTx[0][1] * xTx[1][0]);
                betas[1] = (xTy[1] - xTx[1][0] * betas[0]) / xTx[1][1];

                if (std::abs(betas[0]) < 0.01)
                {
                    is_incremental = false;
                }
                else if (betas[0] > 0 && should_be_increasing)
                {
                    is_incremental = false;
                }
                else if (betas[0] < 0 && !should_be_increasing)
                {
                    is_incremental = false;
                }

                if (start_drift_idx == 250)
                {
                    std::cout << start_drift_idx << " " << end_drift_idx << " " << is_incremental << " " << betas[0] << std::endl;
                }

                int new_start_drift_idx = lookback_consecutive_drift_idx(rows, start_drift_idx, "group", start_mode);
                int new_end_drift_idx;
                if (rows[end_drift_idx].group != end_mode)
                {
                    new_end_drift_idx = lookback_nearest_drift_idx(rows, end_drift_idx, "group", end_mode);
                }
                new_end_drift_idx = lookahead_consecutive_drift_idx(rows, end_drift_idx, "group", end_mode);

                int diff_mode = std::abs(start_mode - end_mode);
                int threshold = 1;
                if (start_mode == 0 || end_mode == 0)
                {
                    threshold = 2;
                }
                if (diff_mode > threshold)
                {
                    bool recurring_exists_within_range = false;
                    for (const auto &[recurring_start, recurring_end] : recurring_drifts)
                    {
                        if (recurring_start == new_start_drift_idx)
                        {
                            recurring_exists_within_range = true;
                            break;
                        }
                    }

                    if (!recurring_exists_within_range)
                    {
                        sudden_count += 1;
                        drifts.emplace_back(new_start_drift_idx, new_end_drift_idx, "sudden");
                    }
                }
                else
                {
                    incremental_count += 1;
                    drifts.emplace_back(new_start_drift_idx, new_end_drift_idx, "incremental");
                }
            }
        }
    }

    std::cout << "Gradual drifts: " << gradual_count << std::endl;
    std::cout << "Sudden drifts: " << sudden_count << std::endl;
    std::cout << "Incremental drifts: " << incremental_count << std::endl;
    std::cout << "Recurring drifts: " << recurring_count << std::endl;
    // sort drifts by start
    std::sort(drifts.begin(), drifts.end(), [](const std::tuple<int, int, std::string> &a, const std::tuple<int, int, std::string> &b)
              { return std::get<0>(a) < std::get<0>(b); });

    // create out csv path
    fs::path out_csv_path = output_path / "raw_drifts_readonly.csv";
    std::ofstream output_file(out_csv_path);
    output_file << "start,end,type,should_use\n";
    for (const auto &[start, end, type] : drifts)
    {
        output_file << start << "," << end << "," << type << "," << "n"
                                                                    "\n";

        fs::path output_tsv_single_path = output_path / "data" / type / (std::to_string(start) + "_" + std::to_string(end) + ".tsv");
        fs::create_directories(output_tsv_single_path.parent_path());
        std::ofstream output_tsv_single_file(output_tsv_single_path);
        output_tsv_single_file << "index\tvalue\n";
        for (int i = start; i < end; ++i)
        {
            output_tsv_single_file << rows[i].index << "\t" << rows[i].value << "\n";
        }
    }
    // copy raw_drift_readonly to selected_drifts.csv, skip if exist
    fs::copy(out_csv_path, output_path / "selected_drifts.csv", fs::copy_options::skip_existing);
    std::cout << "\nDone! See folder " << output_path << std::endl;
    std::cout << "Raw_processed is for debugging purposes only\n";
    std::cout << "tobe_picked_drifts.csv is for selecting drifts to be used in the next step (replaying). Set column should_use to 'y'\n";
    std::cout << "/data folder contains the drifts in tsv format for plotting\n";
    std::cout << "raw_drifts_readonly.csv contains the drifts in csv format for reference\n";
}
}

TraceFinderV1App::TraceFinderV1App(): App(trace_finder_v1::name, trace_finder_v1::description)
{
}

TraceFinderV1App::~TraceFinderV1App()
{
    indicators::show_console_cursor(true);
}

void TraceFinderV1App::setup_args(CLI::App *app)
{
    parser = create_subcommand(app);
    parser->add_option("-i,--input", input, "Input directory")->required()->check(CLI::ExistingDirectory);
    parser->add_option("-o,--output", output, "Output directory")->required();
    parser->add_option("-m,--metric", metric, "Metric")->required();
    parser->add_option("--stability-threshold", stability_threshold, "Stability threshold")->default_val(14.0);
    parser->add_option("--drift-threshold", drift_threshold, "Drift threshold")->default_val(50.0);
    parser->add_option("--group-threshold", group_threshold, "Group threshold")->default_val(250.0);
    parser->add_option("--group-offset", group_offset, "Group offset")->default_val(50.0);
    parser->add_option("--rolling-window", rolling_window, "Rolling window")->default_val(10);
    output = output / metric;
}

void TraceFinderV1App::setup() {
    output = fs::weakly_canonical(output);
    fs::create_directories(output);
}

void TraceFinderV1App::run([[maybe_unused]] CLI::App *app) {
    // print all arguments
    // std::cout << "out_dir: " << out_dir << std::endl;
    // std::cout << "characteristic_file: " << characteristic_file << std::endl;
    // std::cout << "metric: " << metric << std::endl;

    std::vector<trace_finder_v1::Row> rows = trace_finder_v1::read_csv(input, metric);
    std::cout << "Read " << rows.size() << " rows" << std::endl;
    int index = 0;
    for (auto &row : rows)
    {
        // std::cout << "Processing row " << row.index << std::endl;
        row.group = trace_finder_v1::determine_group(row.value, group_threshold, group_offset);
        int last_window = std::max(0, index - rolling_window);
        std::vector<trace_finder_v1::Row> prev_windows(rows.begin() + last_window, rows.begin() + index);

        // std::vector<Row> prev_windows(rows.end() - std::min(ROLLING_WINDOW, static_cast<int>(rows.size())), rows.end());
        prev_windows.push_back(row);
        row.mode = trace_finder_v1::calc_mode_row(prev_windows);

        if (index == 0)
        {
            row.prev_group = -1;
        }
        else
        {
            trace_finder_v1::Row &prev_row = rows[index - 1];
            row.prev_group = prev_row.group;
        }
        index += 1;
    }

    int num_data = rows.size();

    for (int i = 1; i < num_data; ++i)
    {
        trace_finder_v1::Row &row = rows[i];
        trace_finder_v1::Row &prev_row = rows[i - 1];

        std::vector<trace_finder_v1::Row> next_windows(rows.begin() + i, rows.begin() + std::min(num_data, i + rolling_window));
        row.next_mode = trace_finder_v1::calc_mode_row(next_windows);

        int curr_total_mode = std::accumulate(row.mode.begin(), row.mode.end(), 0,
                                              [](int sum, const std::pair<int, int> &p)
                                              { return sum + p.second; });
        int curr_mode_group = row.mode[row.group];
        int next_mode_group = row.next_mode[row.group];

        if (curr_total_mode < rolling_window)
        {
            row.stability = prev_row.stability + 1;
        }
        else
        {
            if (curr_mode_group > (rolling_window / 2) && next_mode_group > (rolling_window / 2))
            {
                row.stability = prev_row.stability + 1;
            }
            else
            {
                int prev_group_mode = prev_row.mode[prev_row.group];
                std::map<int, int> potential_modes = trace_finder_v1::get_similar_modes(row.mode, row.group, 2, false);
                int sum_potential_modes = std::accumulate(potential_modes.begin(), potential_modes.end(), 0,
                                                          [](int sum, const std::pair<int, int> &p)
                                                          { return sum + p.second; });
                if (prev_group_mode > (rolling_window / 2) || sum_potential_modes > (rolling_window / 2))
                {
                    row.stability = prev_row.stability + 1;
                }
                else
                {
                    row.stability = 0;
                }
            }
        }

        if (row.group != prev_row.group)
        {
            row.stability_group = 0;
        }
        else
        {
            row.stability_group = prev_row.stability_group + 1;
        }
    }

    // std::ofstream output_file(output_path / "raw_processed.csv");
    // output_file << "index\tvalue\tstability\tstability_group\tgroup\tprev_group\tmode\tnext_mode\n";
    // for (const auto &row : rows)
    // {
    //     output_file << row.index << "\t" << row.value << "\t" << row.stability << "\t" << row.stability_group
    //                 << "\t" << row.group << "\t" << row.prev_group << "\t";
    //     // Convert map to string for mode and next_mode
    //     std::ostringstream mode_ss, next_mode_ss;
    //     for (const auto &[key, value] : row.mode)
    //     {
    //         mode_ss << key << ":" << value << " ";
    //     }
    //     for (const auto &[key, value] : row.next_mode)
    //     {
    //         next_mode_ss << key << ":" << value << " ";
    //     }
    //     output_file << mode_ss.str() << "\t" << next_mode_ss.str() << "\n";
    // }

    trace_finder_v1::process(rows, output, stability_threshold, rolling_window);
}
}
