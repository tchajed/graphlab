/*
 * Adapted from pagerank demo application.
 *
 */
#include <vector>
#include <string>
#include <fstream>
#include <sys/time.h>

#include <graphlab.hpp>
// #include <graphlab/macros_def.hpp>

// Global random reset probability
double RESET_PROB = 0.15;

double TOLERANCE = 1.0E-2;

std::ofstream FEATURES_FILE;
const std::string FEATURES_FILE_DELIMITER = ",";

struct iteration_value : public graphlab::IS_POD_TYPE {
  int iteration;
  double value;
  
  iteration_value(int iteration_, double value_) : iteration(iteration_), value(value_) {
  }
  
  iteration_value() : iteration(0), value(0.0) {
  }
};

struct diff_vertex {
  std::vector<iteration_value> iteration_values;
  
  diff_vertex() {
  }
  
  double set_value(int iteration, double new_value) {
    iteration_values.push_back(iteration_value(iteration, new_value));
    return new_value - get_value(iteration - 1);
  }
  
  double get_value() const {
    return iteration_values.back().value;
  }
  
  double get_value(int iteration) const {
    if (iteration <= 0) {
      return 0.0;
    }
    double last_value = 0.0;
    for (std::vector<iteration_value>::const_reverse_iterator rit = iteration_values.rbegin();
         rit != iteration_values.rend();
         rit++) {
      if (rit->iteration > iteration) {
        last_value = rit->value;
      } else if (rit->iteration == iteration) {
        return rit->value;
      } else {
        return last_value;
      }
    }
    return last_value;
  }
  
  double get_diff(int iteration) const {
    return get_value(iteration) - get_value(iteration - 1);
  }
  
  double get_second_diff(int iteration) const {
    return get_diff(iteration) - get_diff(iteration - 1);
  }
  
  std::vector<double> get_history() const {
    std::vector<double> pageranks;
    int last_iteration = -1;
    for (std::vector<iteration_value>::const_iterator it = iteration_values.begin(); it != iteration_values.end(); ++it) {
      int iteration = it->iteration;
      double value = it->value;
      int repetitions = 1;
      if (last_iteration != -1) {
        repetitions = iteration - last_iteration;
        ASSERT_TRUE(repetitions > 0);
      }
      for (int i = 0; i < repetitions; i++) {
        pageranks.push_back(value);
      }
      last_iteration = iteration;
    }
    return pageranks;
  }
  
  void save(graphlab::oarchive& oarc) const {
    oarc << iteration_values.size();
    for (std::vector<iteration_value>::const_iterator it = iteration_values.begin(); it != iteration_values.end(); ++it) {
      oarc << *it;
    }
  }
  
  void load(graphlab::iarchive& iarc) {
    size_t n;
    iarc >> n;
    for (unsigned int i = 0; i < n; i++) {
      iteration_value v;
      iarc >> v;
      iteration_values.push_back(v);
    }
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
  }

  /* The scatter edges depend on whether the pagerank has converged */
  edge_dir_type scatter_edges(icontext_type& context,
                              const vertex_type& vertex) const {
    // In the dynamic case we run scatter on out edges if the
    // tolerance is above bound.
    if(std::fabs(last_change) > TOLERANCE ) {
      return graphlab::OUT_EDGES;
    } else {
      return graphlab::NO_EDGES;
    }
  }

  /* The scatter function just signal adjacent pages */
  void scatter(icontext_type& context, const vertex_type& vertex,
               edge_type& edge) const {
    if(last_change > TOLERANCE || last_change < -TOLERANCE) {
        context.signal(edge.target());
    } else {
      context.signal(edge.target());
    }
  }

  void save(graphlab::oarchive& oarc) const {
  }

  void load(graphlab::iarchive& iarc) {
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
unsigned int MAX_VERTICES_ACCURACY = 1000;

// construct top_final_pageranks from final_pageranks
void get_top_pageranks() {
  std::priority_queue<vertex_pagerank,
  std::vector<vertex_pagerank>,
  compare_pageranks_reversed> top_pageranks;
  for (std::map<unsigned long, double>::const_iterator it =
    final_pageranks.begin(); it != final_pageranks.end(); ++it) {
    std::pair<unsigned long, double> vid_pr = *it;
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

struct feature_aggregator {
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
    int iteration = context.iteration();
    agg.features[1] = vertex.data().get_diff(iteration);
    agg.features[2] = vertex.data().get_value(iteration);
    agg.features[3] = vertex.data().get_second_diff(iteration);
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
    for (std::set<vertex_pagerank>::const_iterator it =
      agg.pagerank_list.begin(); it != agg.pagerank_list.end(); ++it) {
      vertex_pagerank vid_pr = *it;
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
      double normalized_feature = sqrt(agg.features[i] / agg.n);
      context.cout() << normalized_feature;
    }
    context.cout().unsetf(std::ios::floatfield);
    context.cout() << std::setprecision(8);
    context.cout() << std::endl;
    
    if (FEATURES_FILE.is_open()) {
      FEATURES_FILE << context.iteration() << FEATURES_FILE_DELIMITER;
      FEATURES_FILE << ranking_rmse << FEATURES_FILE_DELIMITER;
      for (int i = 0; i < agg.num_features; i++) {
        FEATURES_FILE << sqrt(agg.features[i]) / sqrt(agg.n);
        if (i != agg.num_features - 1) {
          FEATURES_FILE << FEATURES_FILE_DELIMITER;
        }
      }
      FEATURES_FILE << "\n";
    }
  }
  
  void save(graphlab::oarchive& oarc) const {
    for (int i = 0; i < num_features; i++) {
      oarc << features[i];
    }
    oarc << pagerank_list.size();
    for (std::set<vertex_pagerank>::const_iterator it = pagerank_list.begin(); it != pagerank_list.end(); ++it) {
      oarc << *it;
    }
  }
  
  void load(graphlab::iarchive& iarc) {
     for (int i = 0; i < num_features; i++) {
       iarc >> features[i];
     }
    size_t n;
    iarc >> n;
    for (int i = 0; i < n; i++) {
      vertex_pagerank vp;
      iarc >> vp;
      pagerank_list.insert(vp);
    }
  }
};

class performance_monitor {
private:
  graphlab::timer timer;
  std::map<std::string, double> start_times;
  std::map<std::string, double> finish_times;
public:
  performance_monitor() {
    timer.start();
  }
  
  void start(std::string name) {
    start_times.insert(std::pair<std::string, double>(name, timer.current_time()));
  }
  
  double finish(const std::string name) {
    std::map<std::string, double>::iterator start_time = start_times.find(name);
    assert(start_time != start_times.end() && "can only finish intervals that have been started");
    double time = timer.current_time() - start_time->second;
    finish_times.insert(std::pair<std::string, double>(name, time));
    start_times.erase(start_time);
    return time;
  }
  
  void write_times(std::ostream &os) const {
    for (std::map<std::string, double>::const_iterator it = finish_times.begin(); it != finish_times.end(); ++it) {
      std::string name = it->first;
      double time = it->second;
      os << name << "\t" << time << std::endl;
    }
    os << "total" << "\t" << timer.current_time() << std::endl;
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
    s << v.num_out_edges() << "\t";
    double last_pagerank = 0.0;
    std::vector<double> history = v.data().get_history();
    for (std::vector<double>::const_iterator it =
      history.begin(); it != history.end(); ++it) {
      double pagerank = *it;
      // print 0/1 to indicate if vertex is active
      if (std::fabs(pagerank - last_pagerank) < TOLERANCE) {
        s << "0\t";
      } else {
        s << "1\t";
      }
      s << pagerank << "\t";
      last_pagerank = pagerank;
    }
    s << "\n";
    return s.str();
  }
  
  std::string save_edge(graph_type::edge_type e) { return ""; }
};

double pagerank_sum(graph_type::vertex_type v) {
  return v.data().get_value();
}

int main(int argc, char** argv) {
  performance_monitor timer;
  // Initialize control plain using mpi
  graphlab::mpi_tools::init(argc, argv);
  graphlab::distributed_control dc;
  global_logger().set_log_level(LOG_WARNING);

  // Parse command line options -----------------------------------------------
  graphlab::command_line_options clopts("PageRank algorithm.");
  std::string graph_dir;
  std::string format = "snap";
  std::string exec_type = "synchronous";
  std::string pagerank_input_prefix;
  bool save_history = false;
  clopts.attach_option("graph", graph_dir,
                       "The graph prefix.");
  clopts.add_positional("graph");
  clopts.attach_option("format", format,
                        "The graph file format");
  clopts.attach_option("pagerank_prefix", pagerank_input_prefix,
                       "input true pageranks");
  clopts.attach_option("engine", exec_type,
                       "The engine type: synchronous or asynchronous");
  clopts.attach_option("tol", TOLERANCE,
                       "The permissible change at convergence.");
  double feature_period = 0.5;
  clopts.attach_option("feature_period", feature_period,
                       "how frequently to compute features");
  clopts.attach_option("save_history", save_history,
                       "whether to save the full history");
  clopts.attach_option("max_vertices", MAX_VERTICES_ACCURACY,
                        "max number of vertices to use for accuracy computation");
  std::string output;
  clopts.attach_option("output", output,
                       "output directory");

  if(!clopts.parse(argc, argv)) {
    dc.cout() << "Error in parsing command line arguments." << std::endl;
    return EXIT_FAILURE;
  }
  
  std::string pagerank_prefix;
  std::string history_prefix;
  std::string features_fname;
  std::string timing_fname;
  if (!output.empty()) {
    pagerank_prefix = output + "/pagerank";
    if (save_history) {
      history_prefix = output + "/history";
    }
    features_fname = output + "/features.csv";
    timing_fname = output + "/timing.tsv";
    if (pagerank_input_prefix.empty()) {
      pagerank_input_prefix = pagerank_prefix;
    }
    boost::filesystem::create_directories(output);
  }

  clopts.get_engine_args().set_option("use_cache", false);

  // Build the graph ----------------------------------------------------------
  timer.start("graph loading");
  graph_type graph(dc, clopts);
  dc.cout() << "Loading graph in format: " << format << std::endl;
  graph.load_format(graph_dir, format);
  // must call finalize before querying the graph
  graph.finalize();
  dc.cout() << "#vertices: " << graph.num_vertices()
            << " #edges:" << graph.num_edges() << std::endl;
  dc.cout() << "loaded graph in " << timer.finish("graph loading") << "s" << std::endl;
  
  // load the true pagerank values if provided
  typedef std::pair<unsigned long, double> pagerank_pair;
  if (!pagerank_input_prefix.empty()) {
    // code adopted from graphlab::load_direct_from_posixfs
    std::string directory_name; std::string original_path(pagerank_prefix);
    boost::filesystem::path path(pagerank_input_prefix);
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
    if (pagerank_files.empty()) {
      logstream(LOG_WARNING) << "no pagerank files match prefix" << std::endl;
    }
    for (std::vector<std::string>::const_iterator it =
         pagerank_files.begin(); it !=pagerank_files.end(); ++it) {
      std::string file = *it;
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
    logstream(LOG_INFO) << "total pageranks loaded: " << final_pageranks.size() << std::endl;
    if (graph.num_vertices() != final_pageranks.size()) {
      logstream(LOG_WARNING) << "pageranks loaded (" << final_pageranks.size() << ") != graph size, discarding pageranks" << std::endl;
      final_pageranks.clear();
    }
  }
  
  // Prepare feature output file
  if (!features_fname.empty()) {
    FEATURES_FILE.open(features_fname.c_str(), std::ios::trunc);
    FEATURES_FILE << "iter" << ",rank_e";
    for (int i = 0; i < feature_aggregator::num_features; i++) {
      FEATURES_FILE << FEATURES_FILE_DELIMITER << feature_aggregator::feature_names[i];
    }
    FEATURES_FILE << std::endl;
  }
  
  // Initialize the vertex data
  graph.transform_vertices(init_vertex);

  // Running The Engine -------------------------------------------------------
  timer.start("engine");
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
  timer.finish("engine");


  // Save the final graph -----------------------------------------------------
  // skip if prefix is empty or we loaded final pageranks (which are assumed correct)
  if (!pagerank_prefix.empty() && (final_pageranks.empty() ||
                                   pagerank_prefix != pagerank_input_prefix)) {
    graph.save(pagerank_prefix, pagerank_writer(),
               true,     // gzip
               true,     // save vertices
               false);   // do not save edges
  }
  
  if (!history_prefix.empty()) {
    graph.save(history_prefix, vertex_history_writer(),
               // gzip, save vertices, don't save edges
               true,
               true,
               false);
  }
  
  double totalpr = graph.map_reduce_vertices<double>(pagerank_sum);
  dc.cout() << "Totalpr = " << totalpr << "\n";
  
  if (FEATURES_FILE.is_open()) {
    FEATURES_FILE.close();
  }
  
  dc.cout() << "finalizing MPI..." << std::endl;
  
  // Tear-down communication layer and quit -----------------------------------
  graphlab::mpi_tools::finalize();
  
  std::cout << "execution times:" << std::endl;
  timer.write_times(std::cout);
  std::ofstream timing_file(timing_fname.c_str(), std::ios::trunc);
  timer.write_times(timing_file);
  timing_file.close();
  return EXIT_SUCCESS;
} // End of main


// We render this entire program in the documentation
