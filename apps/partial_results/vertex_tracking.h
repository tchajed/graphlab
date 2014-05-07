/**
 * vertex_tracking
 *
 * Wrappers for vertex data types to track a history of values
 */

#include <graphlab.hpp>


template <typename T>
struct iteration_value : public graphlab::IS_POD_TYPE {
  int iteration;
  T value;
  
  iteration_value(int iteration_, T value_) : iteration(iteration_), value(value_) {
  }
  
  iteration_value() : iteration(0), value(T()) {
  }
};

template <typename T>
struct diff_vertex {
  std::vector<iteration_value<T> > iteration_values;
  
  diff_vertex() {
  }
  
  T set_value(int iteration, T new_value) {
    iteration_values.push_back(iteration_value<T>(iteration, new_value));
    return new_value - get_value(iteration - 1);
  }
  
  T get_value() const {
    return iteration_values.back().value;
  }
  
  T get_value(int iteration) const {
    if (iteration <= 0) {
      return T();
    }
    T last_value = T();
    for (typename std::vector<iteration_value<T> >::const_reverse_iterator rit = iteration_values.rbegin();
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
  
  T get_diff(int iteration) const {
    return get_value(iteration) - get_value(iteration - 1);
  }
  
  T get_second_diff(int iteration) const {
    return get_diff(iteration) - get_diff(iteration - 1);
  }
  
  std::vector<T> get_history() const {
    std::vector<T> pageranks;
    int last_iteration = -1;
    for (typename std::vector<iteration_value<T> >::const_iterator it = iteration_values.begin(); it != iteration_values.end(); ++it) {
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
    for (typename std::vector<iteration_value<T> >::const_iterator it = iteration_values.begin(); it != iteration_values.end(); ++it) {
      oarc << *it;
    }
  }

  void load(graphlab::iarchive& iarc) {
    size_t n;
    iarc >> n;
    for (unsigned int i = 0; i < n; i++) {
      iteration_value<T> v;
      iarc >> v;
      iteration_values.push_back(v);
    }
  }
  
};
