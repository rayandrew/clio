#include "utils.hpp"

#include <algorithm>
#include <string>
#include <random>

#include <fmt/format.h>
#include <fmt/std.h>

#include <mp-units/format.h>

#include <trace-utils/logger.hpp>
#include <trace-utils/exception.hpp>

#include <trace-utils/exception.hpp>
#include <trace-utils/logger.hpp>

namespace trace_utils {
namespace internal {
static fs::path path_dir_exe = "invalid";
static fs::path path_exe = "invalid";
    
fs::path get_exe_path() {
    if (path_exe == "invalid") {
        path_exe = fs::canonical("/proc/self/exe");
    }
    return path_exe;
}

fs::path get_dir_exe_path() {
    if (path_dir_exe == "invalid") {
        path_dir_exe = get_exe_path().parent_path();
    }
    return path_dir_exe;
}
    
std::string clean_control_characters(std::string_view sv) {
    std::string cleaned;
    cleaned.reserve(sv.size());
    std::copy_if(sv.begin(), sv.end(), std::back_inserter(cleaned), [](unsigned char c) {
        return !std::iscntrl(c) || c == '\n';
        // return std::isprint(c) || c == '\n' || c == ',';
    });
    return cleaned;
}


constexpr std::string_view tar_magic_str = "ustar";
constexpr std::size_t tar_magic_size = tar_magic_str.size();
    
bool is_tar_file(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw Exception(fmt::format("Cannot open file {}", path));
    }

    file.seekg(257, std::ios::beg);

    if (!file) {
        // throw Exception("Could not seek to position 257 of file: " + path);
        return false;
    }

    char buffer[tar_magic_size];
    file.read(buffer, tar_magic_size);
    file.close();

    return std::string_view{buffer}.compare(tar_magic_str) == 0;
}

bool is_gz_file(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw Exception(fmt::format("Cannot open file {}", path));
    }

    // Read the first two bytes
    char buffer[2];
    file.read(buffer, 2);
    file.close();

    // Check if the first two bytes match the gzip header
    return buffer[0] == '\x1F' && buffer[1] == '\x8B';
}

// bool is_tar_gz_file(const fs::path& path) {    
//     std::ifstream file(path, std::ios::binary);
//     if (!file) {
//         throw Exception(fmt::format("Cannot open file {}", path));
//     }

//     // Read the first two bytes
//     char buffer[2];
//     file.read(buffer, 2);

//     // Check if the first two bytes match the gzip header
//     if (!(buffer[0] == '\x1F' && buffer[1] == '\x8B')) {
//         log()->info("Err here");
//         // not gzip
//         return false;
//     }

//     // Skip the gzip header (10 bytes minimum)
//     file.seekg(10, std::ios::beg);

//     if (!file) {
//         log()->info("Err here 2");
//         // throw Exception("Could not seek to position 10 of file: " + path);
//         return false;
//     }

//     // Read the "ustar" magic string
//     file.seekg(257, std::ios::cur);

//     if (!file) {
//         log()->info("Err here 3");
//         return false;
//         // throw Exception("Could not seek to position 257 of file: " + path);
//     }

//     char tar_buffer[tar_magic_size];
//     file.read(tar_buffer, tar_magic_size);
//     file.close();
    
//     log()->info("Here 4, buffer=\"{}\"", tar_buffer);
    

//     return std::string_view{buffer}.compare(tar_magic_str) == 0;
// }

bool is_delimited_file(const fs::path& path, char delimiter, std::size_t num_check_lines) {
    std::ifstream file(path);
    if (!file) {
        throw Exception(fmt::format("Cannot open file {}", path));
    }

    std::string line;
    std::vector<size_t> delimiter_counts;

    while (num_check_lines-- > 0 && std::getline(file, line)) {
        size_t delimiter_count = std::count(line.begin(),
                                            line.end(),
                                            delimiter);
        delimiter_counts.push_back(delimiter_count);
    }
    file.close();

    if (delimiter_counts.size() < 2) {
        return false;
    }

    std::size_t first_line_delimiters = delimiter_counts[0];
    for (std::size_t i = 1; i < delimiter_counts.size(); ++i) {
        if (delimiter_counts[i] != first_line_delimiters) {
            return false;
        }
    }

    return true;
}
} // namespace internal

namespace utils {
// CC-BY-SA 4.0
// Taken from https://stackoverflow.com/a/50556436/2418586
std::string random_string(std::size_t length) {
    std::mt19937 generator{std::random_device{}()};
   //modify range according to your need "A-Z","a-z" or "0-9" or whatever you need.
    std::uniform_int_distribution<int> distribution{'a', 'z'};

    std::string rand_str(length, '\0');
    for(auto& dis: rand_str)
        dis = distribution(generator);

    return rand_str;
}
} // namespace utils
} // namespace trace_utils
