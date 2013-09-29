#ifndef __VERTEX_COLLECTOR_H
#define __VERTEX_COLLECTOR_H

#include <graphlab/serialization/is_pod.hpp>
#include <vector>
#include <map>

template <typename vertex_program, typename graph_type>
class vertex_collector : graphlab::IS_POD_TYPE {
  typedef typename graph_type::vertex_data_type vertex_data_type;
  typedef typename vertex_program::icontext_type context_type;
public:
  struct vertex_info : graphlab::IS_POD_TYPE {
    vertex_data_type data;
    //int in_edges;
    //int out_edges;
  };

  vertex_collector<vertex_program, graph_type>
  operator+=(const vertex_collector<vertex_program, graph_type>& other) {
    if (other.features.size() > 1) {
      std::cout << "adding " << other.features.size() << std::endl;
    }
    features.insert(other.features.begin(), other.features.end());
    return *this;
  }
  
  static vertex_collector<vertex_program, graph_type>
  map(context_type& context,
      const typename graph_type::vertex_type& vertex) {
    vertex_collector coll;
    vertex_info info;
    info.data = vertex.data();
    //info.in_edges = vertex.num_in_edges();
    //info.out_edges = vertex.num_out_edges();
    coll.features.insert(std::pair<unsigned long, vertex_info>(vertex.id(), info));
    return coll;
  }
  
  std::map<unsigned long, vertex_info> get_features() {
    return features;
  }
private:
  
  std::map<unsigned long, vertex_info> features;
  
};

#endif
