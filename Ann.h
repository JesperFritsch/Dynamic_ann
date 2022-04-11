#include <vector>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <fstream>


static double get_random(void);
static void init_random(void);
#define e 3.718281828

class Ann
{
    public:
        Ann(uint32_t inputs, uint32_t width, uint32_t height, uint32_t outputs, double leak, std::string func);
        void train(void);
        bool set_train_data(std::vector<std::vector<double>> xtrain, std::vector<std::vector<double>> ytrain);
        bool set_train_data(std::string filepath);
        void print_train(void);
        bool predict(std::vector<double>& input);
        void set_learning_rate(double learning_rate);
        void set_epochs(uint32_t epochs);
        void print_hidden(void);
        void print_hidden_error(void);
        void print_network(bool weights);
        std::vector<double> predicted;
        void save(std::ofstream& of);
        void load(std::ifstream& inf);
    private:
        uint32_t epochs;
        uint32_t width;
        uint32_t height;
        double learning_rate;
        uint64_t training_time;
        double leak;
        double (Ann::*activation)(double);
        double (Ann::*activation_d)(double);
        std::vector<std::vector<double>> x_train;
        std::vector<std::vector<double>> y_train;
        std::vector<uint32_t> training_order;
        std::vector<double> hidden_error, output_error, input, output, hidden, hidden_bias, output_bias;
        std::vector<std::vector<double>> hidden_weights, input_weights, output_weights;
        void set_activation(std::string func);
        void feed_forward();
        void print_progress(uint32_t current);
        void back_propagate(std::vector<double>& expected);
        double relu(double sum);
        double derived_relu(double sum);
        void setup(void);
        void set_input(std::vector<double>& input);
        void reset_hidden(void);
        void shuffle();
        void optimize(void);
        double sigmoid(double sum);
        double derived_sigmoid(double sum);
        double tanh(double sum);
        double derived_tanh(double sum);

};