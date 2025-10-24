#include "llama.h"
#include <iostream>

int main(int argc, char** argv) {
  llama_backend_init();
  llama_model_params params = llama_model_default_params();
  llama_model* model = llama_model_load_from_file(argv[1], params);

  if (!model) {
    std::cerr << "model: failed to load model" << std::endl;
    return 1;
  }

  std::cout << "model: load success"<< std::endl;
  std::cout << "model: tensor inspection" << std::endl;

  llama_model_free(model); 
  llama_backend_free();

  return 0;
}
