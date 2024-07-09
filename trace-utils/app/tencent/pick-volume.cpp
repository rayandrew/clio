#include "pick-volume.hpp"

#include <cstdio>
#include <algorithm>

#include <archive.h>
#include <archive_entry.h>

#include <fmt/std.h>
#include <glob/glob.h>
#include <natural_sort.hpp>
#include <oneapi/tbb.h>
#include <csv2/mio.hpp>
#include <csv2/writer.hpp>

#include <trace-utils/logger.hpp>
#include <trace-utils/trace/tencent.hpp>
#include <trace-utils/utils.hpp>

namespace trace_utils::app::tencent {
namespace pick_volume {
const char* name = "pick-volume";
const char* description = "Tencent Pick Volume";
}
    
PickVolumeApp::PickVolumeApp(): App(pick_volume::name, pick_volume::description) {

}

void PickVolumeApp::setup_args(CLI::App *app) {
    parser = create_subcommand(app);
    parser->add_option("-i,--input", input, "Input directory")->required()->check(CLI::ExistingDirectory);
    parser->add_option("-o,--output", output, "Output directory")->required(); 
    parser->add_option("-v,--volume", volume, "Choose volume")->required();
}

void PickVolumeApp::setup() {
    tmp_dir_path = fs::temp_directory_path() / fmt::format("{}-{}", pick_volume::name, utils::random_string(50));
    log()->info("Creating temporary directory", tmp_dir_path);
    fs::create_directories(tmp_dir_path);
}

void PickVolumeApp::run([[maybe_unused]] CLI::App* app) {
    auto output_path = fs::weakly_canonical(output);
    fs::create_directories(output_path);
    
    auto input_path = fs::canonical(input) / "*.tgz";
    log()->info("Globbing over {}", input_path);
    auto paths = glob::glob(input_path);
    std::sort(paths.begin(), paths.end(), SI::natural::compare<std::string>);

    oneapi::tbb::parallel_for_each(paths.cbegin(), paths.cend(), [&](const auto& path) {
        using namespace csv2;
        log()->info("Path: {}", path);

        auto stem_path = path.stem();
        auto out_path = fs::weakly_canonical(output_path / stem_path);
        auto archive_file_path = out_path.replace_extension(".tgz");

        auto temp_path = (tmp_dir_path / utils::random_string(50)).replace_extension(".csv");

        log()->debug("Processing path {} to temporary path {}", path, temp_path);

        std::ofstream stream(temp_path);        
        Writer<delimiter<','>> writer(stream);
        trace_utils::trace::TencentTrace trace(path);
        std::vector<std::string> data;
        trace.raw_stream([&](const auto& item) {
            auto buff = item.to_vec();
            writer.write_row(buff);
        }, [&](const auto& item) {
            return item.volume == volume;
        });
        stream.close();

        log()->debug("Writing filtered volume {} from {} to {}", volume, temp_path, archive_file_path);

        struct archive *a;
        struct stat st;
        struct archive_entry *entry;

        a = archive_write_new();
        defer { archive_write_free(a); };
        archive_write_add_filter_gzip(a);
        archive_write_set_format_pax_restricted(a);
        archive_write_open_filename(a, archive_file_path.c_str());
        defer { archive_write_close(a); };

        entry = archive_entry_new();
        defer { archive_entry_free(entry); };

        stat(temp_path.c_str(), &st);

        archive_entry_set_pathname(entry, stem_path.c_str());
        archive_entry_set_filetype(entry, AE_IFREG);
        archive_entry_set_perm(entry, 0644);
        archive_entry_copy_stat(entry, &st);
        archive_write_header(a, entry);

        auto file = mio::mmap_source(temp_path.string());
        if (!file.is_open() || !file.is_mapped()) {
            throw Exception(fmt::format("Cannot mmap file {}", temp_path));
        }
        archive_write_data(a, file.data(), file.mapped_length());
    });

    log()->info("Removing temporary directory", tmp_dir_path);
    fs::remove_all(tmp_dir_path);
}
} // namespace trace_utils::app::tencent
