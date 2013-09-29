/*
 * Adapted from pagerank demo application.
 *
 */
#include <vector>
#include <string>
#include <fstream>

#include <graphlab.hpp>
// #include <graphlab/macros_def.hpp>

// Global random reset probability
double RESET_PROB = 0.15;

double TOLERANCE = 1.0E-2;

size_t ITERATIONS = 0;

bool USE_DELTA_CACHE = false;

struct diff_vertex : public graphlab::IS_POD_TYPE {
private:
  std::vector< std::pair<int, double> > values;
  
public:
  diff_vertex() {
  }
  
  double set_value(int iteration, double new_value) {
    values.push_back(std::pair<int, double>(iteration, new_value));
    return get_diff();
  }
  
  double get_value() const {
    return values.back().second;
  }
  
  double get_diff() const {
    size_t n = values.size();
    if (n <= 1) {
      return 0.0;
    }
    return values[n - 1].second - values[n - 2].second;
  }
  
  double get_second_diff() const {
    size_t n = values.size();
    if (n <= 2) {
      if (n <= 1) {
        return 0.0;
      }
      return get_diff();
    }
    return values[n - 1].second - 2*values[n - 2].second + values[n - 3].second;
  }
  
  std::vector<double> get_history() const {
    std::vector<double> pageranks;
    int last_iteration = -1;
    for (std::pair<int, double> iteration_value : values) {
      int repetitions = 1;
      if (last_iteration != -1) {
        repetitions = iteration_value.first - last_iteration;
        ASSERT_TRUE(repetitions > 0);
      }
      for (int i = 0; i < repetitions; i++) {
        pageranks.push_back(iteration_value.second);
      }
      last_iteration = iteration_value.first;
    }
    return pageranks;
  }
};


// The vertex data is just the pagerank value (a double), with some difference tracking
typedef diff_vertex vertex_data_type;

// There is no edge data in the pagerank application
typedef graphlab::empty edge_data_type;

// The graph type is determined by the vertex and edge data types
typedef graphlab::distributed_graph<vertex_data_type, edge_data_type> graph_type;

/*
 * A simple function used by graph.transform_vertices(init_vertex);
 * to initialize the vertes data.
 */
void init_vertex(graph_type::vertex_type& vertex) {
  vertex.data().set_value(-1, 1.0);
}


/*
 * The factorized page rank update function extends ivertex_program
 * specifying the:
 *
 *   1) graph_type
 *   2) gather_type: double (returned by the gather function). Note
 *      that the gather type is not strictly needed here since it is
 *      assumed to be the same as the vertex_data_type unless
 *      otherwise specified
 *
 * In addition ivertex program also takes a message type which is
 * assumed to be empty. Since we do not need messages no message type
 * is provided.
 *
 * pagerank also extends graphlab::IS_POD_TYPE (is plain old data type)
 * which tells graphlab that the pagerank program can be serialized
 * (converted to a byte stream) by directly reading its in memory
 * representation.  If a vertex program does not extend
 * graphlab::IS_POD_TYPE it must implement load and save functions.
 */
class pagerank :
  public graphlab::ivertex_program<graph_type, double> {

  double last_change;
public:

  /**
   * Gather only in edges.
   */
  edge_dir_type gather_edges(icontext_type& context,
                              const vertex_type& vertex) const {
    return graphlab::IN_EDGES;
  } // end of Gather edges


  /* Gather the weighted rank of the adjacent page   */
  double gather(icontext_type& context, const vertex_type& vertex,
               edge_type& edge) const {
    return (edge.source().data().get_value() / edge.source().num_out_edges());
  }

  /* Use the total rank of adjacent pages to update this page */
  void apply(icontext_type& context, vertex_type& vertex,
             const gather_type& total) {

    const double newval = (1.0 - RESET_PROB) * total + RESET_PROB;
    last_change = vertex.data().set_value(context.iteration(), newval);
    if (ITERATIONS) context.signal(vertex);
  }

  /* The scatter edges depend on whether the pagerank has converged */
  edge_dir_type scatter_edges(icontext_type& context,
                              const vertex_type& vertex) const {
    // If an iteration counter is set then
    if (ITERATIONS) return graphlab::NO_EDGES;
    // In the dynamic case we run scatter on out edges if the we need
    // to maintain the delta cache or the tolerance is above bound.
    if(USE_DELTA_CACHE || std::fabs(last_change) > TOLERANCE ) {
      return graphlab::OUT_EDGES;
    } else {
      return graphlab::NO_EDGES;
    }
  }

  /* The scatter function just signal adjacent pages */
  void scatter(icontext_type& context, const vertex_type& vertex,
               edge_type& edge) const {
    if(USE_DELTA_CACHE) {
      context.post_delta(edge.target(), last_change);
    }

    if(last_change > TOLERANCE || last_change < -TOLERANCE) {
        context.signal(edge.target());
    } else {
      context.signal(edge.target()); //, std::fabs(last_change));
    }
  }

  void save(graphlab::oarchive& oarc) const {
    // If we are using iterations as a counter then we do not need to
    // move the last change in the vertex program along with the
    // vertex data.
    if (ITERATIONS == 0) oarc << last_change;
  }

  void load(graphlab::iarchive& iarc) {
    if (ITERATIONS == 0) iarc >> last_change;
  }

}; // end of factorized_pagerank update functor

typedef std::pair<unsigned long, double> vertex_pagerank;

struct compare_pageranks {
  bool operator() (vertex_pagerank a, vertex_pagerank b) {
    return a.second < b.second;
  }
};

struct compare_pageranks_reversed {
  bool operator() (vertex_pagerank a, vertex_pagerank b) {
    return b.second < a.second;
  }
};

std::map<unsigned long, double> final_pageranks;
// map from vid to ranking (within the top)
std::map<unsigned long, unsigned int> top_final_pageranks;
typedef std::pair<unsigned long, unsigned int> vertex_rank;

// max number of vertices to use in accuracy computation
int MAX_VERTICES_ACCURACY = 1000;

// construct top_final_pageranks from final_pageranks
void get_top_pageranks() {
  std::priority_queue<vertex_pagerank,
  std::vector<vertex_pagerank>,
  compare_pageranks_reversed> top_pageranks;
  for (vertex_pagerank vid_pr : final_pageranks) {
    if (top_pageranks.size() >= MAX_VERTICES_ACCURACY &&
        vid_pr.second <= top_pageranks.top().second) {
      continue;
    }
    top_pageranks.push(vid_pr);
    while (top_pageranks.size() > MAX_VERTICES_ACCURACY) {
      top_pageranks.pop();
    }
  }
  unsigned int rank = 0;
  while (!top_pageranks.empty()) {
    vertex_pagerank vid_pr = top_pageranks.top();
    top_final_pageranks.insert(vertex_rank(vid_pr.first, rank));
    top_pageranks.pop();
    rank++;
  }
}

struct feature_aggregator : public graphlab::IS_POD_TYPE {
  typedef pagerank::icontext_type context_type;
  double features[4];
  std::set<vertex_pagerank, compare_pageranks> pagerank_list;
  static const std::string feature_names[4];
  static const int num_features = sizeof(feature_names)/sizeof(feature_names[0]);
  double n;
  
  feature_aggregator& operator+=(const feature_aggregator& other) {
    for (int i = 0; i < num_features; i++) {
      features[i] += other.features[i];
    }
    pagerank_list.insert(other.pagerank_list.begin(), other.pagerank_list.end());
    std::set<vertex_pagerank, compare_pageranks>::iterator it = pagerank_list.begin();
    while (pagerank_list.size() > MAX_VERTICES_ACCURACY) {
      pagerank_list.erase(it++);
    }
    n += other.n;
    return *this;
  }
  
  static feature_aggregator map(context_type& context,
    const graph_type::vertex_type& vertex) {
    feature_aggregator agg;
    if (final_pageranks.empty()) {
      agg.features[0] = 0.0;
    } else {
      double true_pagerank = final_pageranks.at(vertex.id());
      agg.features[0] = vertex.data().get_value() - true_pagerank;
    }
    agg.features[1] = vertex.data().get_diff();
    agg.features[2] = vertex.data().get_value();
    agg.features[3] = vertex.data().get_second_diff();
    agg.pagerank_list.insert(vertex_pagerank(vertex.id(), vertex.data().get_value()));
    // square features
    for (int i = 0; i < num_features; i++) {
      agg.features[i] *= agg.features[i];
    }
    agg.n = 1;
    return agg;
  }
  
  static void finalize(context_type& context, feature_aggregator agg) {
    // compute ranking error as a final features
    double ranking_rmse = 0.0;
    int rank = 0;
    for (vertex_pagerank vid_pr : agg.pagerank_list) {
      int true_rank;
      if (top_final_pageranks.find(vid_pr.first) == top_final_pageranks.end()) {
        true_rank = -1;
      } else {
        true_rank = top_final_pageranks.at(vid_pr.first);
      }
      int rank_error = rank - true_rank;
      ranking_rmse += rank_error * rank_error;
      rank++;
    }
    ranking_rmse /= top_final_pageranks.size();
    ranking_rmse = sqrt(ranking_rmse);
    
    context.cout() << std::scientific << std::setprecision(3);
    context.cout() << "[iter=" << std::setw(2) << context.iteration() << "]";
    context.cout() << " rank_e: " << ranking_rmse;
    for (int i = 0; i < agg.num_features; i++) {
      context.cout() << " " + agg.feature_names[i] << ": ";
      double normalized_feature = sqrt(agg.features[i]) / sqrt(agg.n);
      context.cout() << normalized_feature;
    }
    context.cout().unsetf(std::ios::floatfield);
    context.cout() << std::setprecision(8);
    
    
    context.cout() << std::endl;
  }
};

const std::string feature_aggregator::feature_names[] =
  {"rmse", "d/di", "size", "d^2/di^2"};

/*
 * We want to save the final graph so we define a write which will be
 * used in graph.save("path/prefix", pagerank_writer()) to save the graph.
 */
struct pagerank_writer {
  std::string save_vertex(graph_type::vertex_type v) {
    std::stringstream strm;
    strm << v.id() << "\t" << v.data().get_value() << "\n";
    return strm.str();
  }
  std::string save_edge(graph_type::edge_type e) { return ""; }
}; // end of pagerank writer

struct vertex_history_writer {
  std::string save_vertex(graph_type::vertex_type v) {
    std::stringstream s;
    s << v.id() << "\t";
    s << v.num_in_edges() << "\t";
    double last_pagerank = 0.0;
    for (double pagerank : v.data().get_history()) {
      // print 0/1 to indicate if vertex is active
      if (std::fabs(pagerank - last_pagerank) < TOLERANCE) {
        s << "0\t";
      } else {
        s << "1\t";
      }
      s << pagerank << "\t";
      last_pagerank = pagerank;
    }
    return s.str();
  }
  
  std::string save_edge(graph_type::edge_type e) { return ""; }
};

double pagerank_sum(graph_type::vertex_type v) {
  return v.data().get_value();
}

int main(int argc, char** argv) {
  // Initialize control plain using mpi
  graphlab::mpi_tools::init(argc, argv);
  graphlab::distributed_control dc;
  global_logger().set_log_level(LOG_WARNING);

  // Parse command line options -----------------------------------------------
  graphlab::command_line_options clopts("PageRank algorithm.");
  std::string graph_dir;
  std::string format = "adj";
  std::string exec_type = "synchronous";
  clopts.attach_option("graph", graph_dir,
                       "The graph file.  If none is provided "
                       "then a toy graph will be created");
  clopts.add_positional("graph");
  clopts.attach_option("engine", exec_type,
                       "The engine type synchronous or asynchronous");
  clopts.attach_option("tol", TOLERANCE,
                       "The permissible change at convergence.");
  clopts.attach_option("format", format,
                       "The graph file format");
  size_t powerlaw = 0;
  clopts.attach_option("powerlaw", powerlaw,
                       "Generate a synthetic powerlaw out-degree graph. ");
  clopts.attach_option("iterations", ITERATIONS,
                       "If set, will force the use of the synchronous engine"
                       "overriding any engine option set by the --engine parameter. "
                       "Runs complete (non-dynamic) PageRank for a fixed "
                       "number of iterations. Also overrides the iterations "
                       "option in the engine");
  clopts.attach_option("use_delta", USE_DELTA_CACHE,
                       "Use the delta cache to reduce time in gather.");
  std::string saveprefix;
  clopts.attach_option("saveprefix", saveprefix,
                       "If set, will save the resultant pagerank to a "
                       "sequence of files with prefix saveprefix");
  double feature_period = 0.5;
  clopts.attach_option("feature_period", feature_period,
                       "how frequently to compute features");
  std::string pagerank_prefix;
  clopts.attach_option("pagerank_prefix", pagerank_prefix,
                       "prefix of files to load true pagerank from "
                       "(for accuracy computation)");
  std::string history_prefix;
  clopts.attach_option("history_prefix", history_prefix,
                       "prefix of files to save full vertex history to");
  clopts.attach_option("max_vertices", MAX_VERTICES_ACCURACY,
                       "max number of vertices to use for accuracy computation");

  if(!clopts.parse(argc, argv)) {
    dc.cout() << "Error in parsing command line arguments." << std::endl;
    return EXIT_FAILURE;
  }

  // Enable gather caching in the engine
  clopts.get_engine_args().set_option("use_cache", USE_DELTA_CACHE);

  if (ITERATIONS) {
    // make sure this is the synchronous engine
    dc.cout() << "--iterations set. Forcing Synchronous engine, and running "
              << "for " << ITERATIONS << " iterations." << std::endl;
    clopts.get_engine_args().set_option("type", "synchronous");
    clopts.get_engine_args().set_option("max_iterations", ITERATIONS);
    clopts.get_engine_args().set_option("sched_allv", true);
  }

  // Build the graph ----------------------------------------------------------
  graph_type graph(dc, clopts);
  if(powerlaw > 0) { // make a synthetic graph
    dc.cout() << "Loading synthetic Powerlaw graph." << std::endl;
    graph.load_synthetic_powerlaw(powerlaw, false, 2.1, 100000000);
  }
  else if (!graph_dir.empty()) { // Load the graph from a file
    dc.cout() << "Loading graph in format: "<< format << std::endl;
    graph.load_format(graph_dir, format);
  }
  else {
    dc.cout() << "graph or powerlaw option must be specified" << std::endl;
    clopts.print_description();
    return 0;
  }
  // must call finalize before querying the graph
  graph.finalize();
  dc.cout() << "#vertices: " << graph.num_vertices()
            << " #edges:" << graph.num_edges() << std::endl;
  
  // load the true pagerank values if provided
  typedef std::pair<unsigned long, double> pagerank_pair;
  if (!pagerank_prefix.empty()) {
    // code adopted from graphlab::load_direct_from_posixfs
    std::string directory_name; std::string original_path(pagerank_prefix);
    boost::filesystem::path path(pagerank_prefix);
    std::string search_prefix;
    if (boost::filesystem::is_directory(path)) {
      // if this is a directory
      // force a "/" at the end of the path
      // make sure to check that the path is non-empty. (you do not
      // want to make the empty path "" the root path "/" )
      directory_name = path.native();
    } else {
      directory_name = path.parent_path().native();
      search_prefix = path.filename().native();
      directory_name = (directory_name.empty() ? "." : directory_name);
    }
    std::vector<std::string> pagerank_files;
    graphlab::fs_util::list_files_with_prefix(directory_name,
                                              search_prefix, pagerank_files);
    for (std::string file : pagerank_files) {
      std::ifstream in_file(file.c_str(),
                            std::ios_base::in | std::ios_base::binary);
      boost::iostreams::filtering_stream<boost::iostreams::input> fin;
      const bool gzip = boost::ends_with(file, ".gz");
      if (gzip) {
        fin.push(boost::iostreams::gzip_decompressor());
      }
      fin.push(in_file);
      while (fin.good() && !fin.eof()) {
        unsigned long int vid;
        double pagerank;
        fin >> vid;
        fin >> pagerank;
        final_pageranks.insert(pagerank_pair(vid, pagerank));
      }
      fin.pop();
      if (gzip) {
        fin.pop();
      }
    }
    get_top_pageranks();
    dc.cerr() << "total pageranks loaded: " << final_pageranks.size() << std::endl;
  }
  
  // Initialize the vertex data
  graph.transform_vertices(init_vertex);

  // Running The Engine -------------------------------------------------------
  graphlab::omni_engine<pagerank> engine(dc, graph, exec_type, clopts);
  engine.add_vertex_aggregator<feature_aggregator>("feature_calculation",
                                                   feature_aggregator::map,
                                                   feature_aggregator::finalize);
  engine.aggregate_periodic("feature_calculation", feature_period);
  
  engine.signal_all();
  engine.start();
  const double runtime = engine.elapsed_seconds();
  dc.cout() << "Finished Running engine in " << runtime
            << " seconds." << std::endl;


  // Save the final graph -----------------------------------------------------
  if (!saveprefix.empty()) {
    graph.save(saveprefix, pagerank_writer(),
               true,     // gzip
               true,     // save vertices
               false);   // do not save edges
  }
  if (!history_prefix.empty()) {
    graph.save(history_prefix, vertex_history_writer(),
               // no gzip, save vertices, don't save edges
               false,
               true,
               false);
  }
  
  double totalpr = graph.map_reduce_vertices<double>(pagerank_sum);
  std::cout << "Totalpr = " << totalpr << "\n";
  
  // Tear-down communication layer and quit -----------------------------------
  graphlab::mpi_tools::finalize();
  return EXIT_SUCCESS;
} // End of main


// We render this entire program in the documentation
