#include <graphlab.hpp>
#include <graphlab/warp.hpp>
#include <graphlab/engine/warp_engine.hpp>

enum selected_t { SELECTED, NOT_SELECTED };

typedef selected_t vertex_data_type;
typedef selected_t edge_data_type;

typedef graphlab::distributed_graph<vertex_data_type, edge_data_type> graph_type;
typedef graphlab::warp::warp_engine<graph_type> engine_type;

unsigned int num_sampled;
unsigned int target_sample;

// simple init that sets all vertices to selected
void init_vertex(graph_type::vertex_type& vertex) {
  vertex.data() = SELECTED;
}

// init all edges to selected
void init_edge(graph_type::edge_type& edge) {
  edge.data() = SELECTED;
}

struct graph_writer {
  std::string save_vertex(graph_type::vertex_type v) {
    return "";
  }
  
  std::string save_edge(graph_type::edge_type e) {
    if (e.data() == NOT_SELECTED ||
        e.source().data() == NOT_SELECTED ||
        e.target().data() == NOT_SELECTED) {
      return "";
    }
    std::stringstream strm;
    strm << e.source().id() << "\t" << e.target().id() << std::endl;
    return strm.str();
  }
};

bool random_boolean(double prob) {
  return (graphlab::random::rand01() < prob);
}

void signal_neighbor(engine_type::context& context,
                     graph_type::edge_type edge,
                     graph_type::vertex_type other) {
  context.signal(other);
}

void random_forest_update(engine_type::context &context,
                            graph_type::vertex_type vertex) {
  // don't do anything if this vertex is already burned
  if (vertex.data() == NOT_SELECTED) {
    return;
  }
  // immediately stop fires if reached sampling goal
  if (num_sampled < target_sample) {
    return;
  }
  vertex.data() = NOT_SELECTED;
  num_sampled--;
  // TODO: use the fire's death probability here
  if (random_boolean(0.15)) {
    return;
  }
  // spread the fire
  graphlab::warp::broadcast_neighborhood(context,
                                         vertex,
                                         graphlab::OUT_EDGES,
                                         signal_neighbor);
}

struct random_vertex_sampler {
  random_vertex_sampler(double prob) : prob(prob) {
  }
  
  bool operator()(const graph_type::vertex_type& vertex) {
    if (vertex.data() == NOT_SELECTED) {
      return false;
    }
    return graphlab::random::rand01() < prob;
  }
  
private:
  double prob;
};

int main(int argc, char** argv) {
  // Initialize control plain using mpi
  graphlab::mpi_tools::init(argc, argv);
  graphlab::distributed_control dc;
  global_logger().set_log_level(LOG_WARNING);

  graphlab::command_line_options clopts("Graph Sampler");
  std::string graph_dir;
  std::string format;
  std::string output;
  double rampup = 1.0;
  clopts.attach_option("graph", graph_dir,
                       "the input graph prefix");
  clopts.attach_option("format", format,
                       "the input graph format");
  clopts.attach_option("output", output,
                       "output prefix");
  clopts.attach_option("rampup", rampup,
                       "sampling rampup per 10 iterations");
  
  double prob = 0.5;
  clopts.attach_option("prob", prob,
                       "sampling probability");

  if(!clopts.parse(argc, argv)) {
    dc.cerr() << "Error in parsing command line arguments." << std::endl;
    return EXIT_FAILURE;
  }
  
  if (graph_dir.empty() || format.empty()) {
    dc.cerr() << "Must specify graph and format" << std::endl;
    return EXIT_FAILURE;
  }
  
  graph_type graph(dc, clopts);
  graph.load_format(graph_dir, format);
  graph.finalize();
  graph.transform_vertices(init_vertex);
  graph.transform_edges(init_edge);
  
  
  num_sampled = graph.num_vertices();
  target_sample = (unsigned int) ((double) num_sampled * prob);
  dc.cout() << "aiming for " << target_sample << " vertices" << std::endl;
  engine_type engine(dc, graph);
  engine.set_update_function(random_forest_update);
  double prob_per_sampled = 2.0;
  unsigned int iteration = 0;
  while (num_sampled > target_sample) {
    graphlab::vertex_set seed_vset = graph.select(random_vertex_sampler(prob_per_sampled/(double) num_sampled));
    dc.cout() << "starting fires at " << graph.vertex_set_size(seed_vset) << " vertices" << std::endl;
    engine.signal_vset(seed_vset);
    engine.start();
    iteration++;
    // ramp up sampling to ensure completion with low sampling percentage
    if (iteration % 10 == 0) {
      prob_per_sampled += rampup;
    }
  }
  
  dc.cout() << "took " << iteration << " iterations" << std::endl;
  dc.cout() << "sampled " << num_sampled << " vertices" << std::endl;
  dc.cout() << "actual/target " << (double) num_sampled / (double) target_sample << std::endl;
  
  if (!output.empty()) {
    graph.save(output, graph_writer(),
               // gzip, don't save vertices, save edges
               true,
               false,
               true);
  }

  // Tear-down communication layer and quit -----------------------------------
  graphlab::mpi_tools::finalize();
  return EXIT_SUCCESS;
}
