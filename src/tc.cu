#include <vector>
#include <optional>

#include <gunrock/algorithms/tc.hxx>
#include "tc_cpu.hxx"

#include <cxxopts.hpp>

using namespace gunrock;
using namespace memory;

template <typename csr_t>

csr_t sparsify_csr(const csr_t& csr, float p, unsigned int seed = 42) {
    using edge_t = typename csr_t::offset_type;
    using vertex_t = typename csr_t::index_type;

    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(0.0, 1.0);

    csr_t csr_sparse;
    csr_sparse.number_of_rows = csr.number_of_rows;
    csr_sparse.number_of_columns = csr.number_of_columns;
    csr_sparse.row_offsets.resize(csr.number_of_rows + 1);
    
    std::vector<vertex_t> new_column_indices;
    new_column_indices.reserve(csr.number_of_nonzeros);

    csr_sparse.row_offsets[0] = 0;
    for (vertex_t i = 0; i < csr.number_of_rows; ++i) {
        for (edge_t offset = csr.row_offsets[i]; offset < csr.row_offsets[i + 1]; ++offset) {
            if (dis(gen) <= p) {
                new_column_indices.push_back(csr.column_indices[offset]);
            }
        }
        csr_sparse.row_offsets[i + 1] = new_column_indices.size();
    }

    csr_sparse.column_indices = new_column_indices;
    csr_sparse.number_of_nonzeros = new_column_indices.size();

    return csr_sparse;
}

struct parameters_t {
  std::string filename;
  cxxopts::Options options;
  bool validate;
  bool reduce_all_triangles;
  std::optional<float> sparsity; // Use std::optional for sparsity level

  parameters_t(int argc, char** argv)
      : options(argv[0], "Accelerating Approximate Triangle Counting") {
    options.add_options()
      ("help", "Print help")
      ("v,validate", "CPU validation using Node-Iterator",
       cxxopts::value<bool>()->default_value("true"))
      ("m,market", "Matrix file", cxxopts::value<std::string>())
      ("r,reduce",
       "Compute a single triangle count for the entire graph",
       cxxopts::value<bool>()->default_value("true"))
      ("s,sparsify", "Do approximate triangle counting, and set edge sparsification percentage (0-1) (default: false)", 
       cxxopts::value<float>()); 

    // Parse command line arguments
    auto result = options.parse(argc, argv);

    if (result.count("help") || (result.count("market") == 0)) {
      std::cout << options.help({""}) << std::endl;
      std::exit(0);
    }

    filename = result["market"].as<std::string>();
    validate = result["validate"].as<bool>();
    reduce_all_triangles = result["reduce"].as<bool>();
    if (result.count("sparsify")) {
      sparsity = result["sparsify"].as<float>();
    }
  }
};

void test_tc(int num_arguments, char** argument_array) {
  // --
  // Define types

  using vertex_t = uint32_t;
  using edge_t = uint32_t;
  using weight_t = float;
  using count_t = vertex_t;

  using csr_t =
      format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;
  csr_t csr;

  // --
  // IO
  parameters_t params(num_arguments, argument_array);
  gunrock::graph::graph_properties_t properties =
      gunrock::graph::graph_properties_t();

  if (util::is_market(params.filename)) {
    io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
    auto [properties, coo] = mm.load(params.filename);
    if (!properties.symmetric) {
      std::cerr << "Error: input matrix must be of an undirected/symmetric graph" << std::endl;
      exit(1);
    }
    csr.from_coo(coo);
  } else if (util::is_binary_csr(params.filename)) {
    csr.read_binary(params.filename);
  } else {
    std::cerr << "Unknown file format: " << params.filename << std::endl;
    exit(1);
  }

  // --
  // Build graph
  float p = 1.0;
  if (params.sparsity) {
    p = *params.sparsity;
    csr = sparsify_csr(csr, p);
  }
  auto G = graph::build<memory_space_t::device>(properties, csr);

  // --
  // Params and memory allocation

  vertex_t n_vertices = G.get_number_of_vertices();
  thrust::device_vector<count_t> triangles_count(n_vertices, 0);

  // --
  // GPU Run

  std::size_t total_triangles = 0;
  float gpu_elapsed = tc::run(G, params.reduce_all_triangles,
                              triangles_count.data().get(), &total_triangles);

  // --
  // Log

  print::head(triangles_count, 40, "Per-vertex triangle count");
  if (params.sparsity) {
    std::cout << "Sparsity Percentage (p): " << p << std::endl;
    std::cout << "Sparse Triangle Counts (GPU): " << total_triangles << std::endl;
  } else {
    if (params.reduce_all_triangles) {
      std::cout << "Total GPU Triangles : " << total_triangles << std::endl;
    }
  }
  std::cout << "GPU Elapsed Time : " << gpu_elapsed << " (ms)" << std::endl;

  // --
  // CPU validation
  if (params.validate) {
    std::vector<count_t> reference_triangles_count(n_vertices, 0);
    std::size_t reference_total_triangles = 0;

    float cpu_elapsed =
        tc_cpu::run(csr, reference_triangles_count, reference_total_triangles);
    uint32_t n_errors = 0;

    if (params.sparsity) {
      std::cout << "Discrepancy in Sparse Triangle Counts (GPU vs. CPU): " 
                << total_triangles << " != " << reference_total_triangles << std::endl;
      std::cout << "Total Triangle Mismatch (GPU vs. CPU):" 
                << total_triangles / (p * p * p) << " != " 
                << reference_total_triangles / (p * p * p) << std::endl;
      n_errors++;
    } else {
      std::cout << "Discrepancy in Exact Triangle Counts (GPU vs. CPU): " 
                << total_triangles << " != " << reference_total_triangles << std::endl;
      n_errors += util::compare(
          triangles_count.data().get(), reference_triangles_count.data(),
          n_vertices, [](const auto x, const auto y) { return x != y; }, true);
      std::cout << "Number of errors : " << n_errors << std::endl;
    }

    std::cout << "CPU Elapsed Time : " << cpu_elapsed << " (ms)" << std::endl;
  }
}

int main(int argc, char** argv) {
  test_tc(argc, argv);
}
