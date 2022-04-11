#include "header.h"

Ann::Ann(uint32_t inputs, uint32_t width, uint32_t height, uint32_t outputs, double leak, std::string func){
    this->width = width;
    this->height = height;
    this->epochs = 10000;
    this->learning_rate = 0.01;
    this->predicted.resize(outputs);
    this->input.resize(inputs);
    this->hidden.resize(width*height);
    this->output.resize(outputs);
    this->hidden_bias.resize(width*height);
    this->output_bias.resize(outputs);
    this->hidden_error.resize(width*height);
    this->output_error.resize(outputs);
    this->input_weights.resize(inputs, std::vector<double>(height));
    this->hidden_weights.resize((this->hidden.size() - height), std::vector<double>(height));
    this->output_weights.resize(outputs, std::vector<double>(height));
    this->leak = leak;
    setup();
    set_activation(func);
}

void Ann::set_activation(std::string func){
    if(func == "relu"){
        printf("relu\n");
        this->activation = &Ann::relu;
        this->activation_d = &Ann::derived_relu;
    }
    else if(func == "sigmoid"){
        printf("sigmoid\n");
        this->activation = &Ann::sigmoid;
        this->activation_d = &Ann::derived_sigmoid;
    }
    else if(func == "tanh"){
        printf("tanh\n");
        this->activation = &Ann::tanh;
        this->activation_d = &Ann::derived_tanh;
    }
}

void Ann::train(void){
    uint64_t ms_start = get_ms();
    for(register uint32_t i = 0; i < this->epochs; i++){
        print_progress(i);
        shuffle();
        for(register uint32_t j = 0; j < this->x_train.size(); j++){
            uint32_t index = training_order[j];
            set_input(this->x_train[index]);
            feed_forward();
            back_propagate(y_train[index]);
            optimize();

        }
    }
    uint64_t ms_end = get_ms();
    uint64_t ms_total = ms_end - ms_start;
    print_time_from_ms(ms_total);

    return;
}

void Ann::print_network(bool weights){
    for(uint32_t i = 0; i < this->height; i++){
        if(i < this->input.size()){
            printf("in: (%5.2f)  ", this->input[i]);
        }
        for(uint32_t j = 0; j < 2; j++){
            printf("\r\t\t");
            if(j == 0){
                for(uint32_t k = 0; k < this->hidden.size(); k += this->height){
                    printf("b[%d]: %5.2f | ",(i + k), this->hidden_bias[i + k]);
                }
                if(i < this->output.size()){
                    printf("  outb: (%5.2f)", this->output_bias[i]);
                }
                printf("\n");
            }
            else{
                for(uint32_t k = 0; k < this->hidden.size(); k += this->height){
                    printf("h[%d]: %5.2f | ",(i + k), this->hidden[i + k]);
                }
                if(i < this->output.size()){
                    printf("   out: (%5.2f)", this->output[i]);
                }
                printf("\n\n");
            }
        }
    }
    if(weights){
        printf("_______Weights_______\n\n");
        for(uint32_t i = 0; i < this->height; i++){
            for(uint32_t j = 0; j < this->height; j++){
                printf("\t");
                if(i < this->input.size()){
                    printf("iw: %5.2f | ", this->input_weights[i][j]);
                }
                printf("\r\t\t\t");
                for(uint32_t k = 0; k < (this->hidden.size() - this->height); k += this->height){
                    printf("hw[%d][%d]: %5.2f | ",(i + k), j, this->hidden_weights[i + k][j]);
                }
                if(i < this->output.size()){
                    printf("ow: %5.2f", this->output_weights[i][j]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }
}

void Ann::print_hidden(void){
    for(uint32_t i = 0; i < this->height; i++){
        for(uint32_t j = 0; j < this->width; j++){
            printf("%.3f | ", this->hidden[i + (this->height * j)]);
        }
        printf("\n");
    }
}

void Ann::print_hidden_error(void){
    for(uint32_t i = 0; i < this->height; i++){
        for(uint32_t j = 0; j < this->width; j++){
            printf("%.3f | ", this->hidden_error[i + (this->height * j)]);
        }
        printf("\n");
    }
}

void Ann::print_train(void){
    for(uint32_t i = 0; i < 10; i++){
	printf("%f %f %f\n", this->x_train[i][0], this->x_train[i][1], this->x_train[i][2]);
    printf("%f %f %f\n", this->x_train[i][3], this->x_train[i][4], this->x_train[i][5]);
    printf("%f %f %f\n", this->x_train[i][6], this->x_train[i][7], this->x_train[i][8]);
    printf("y: %f\n", this->y_train[i][0]);
	printf("\n");
    }
}

void Ann::print_progress(uint32_t current){
    uint8_t step = ceil((double)this->epochs / 100);
    if((current % step) == 0 || current == 1){
        char bar[40] = "";
        uint8_t percents;
        for(uint32_t i = 0; i < (40 * ((double)current / this->epochs)); i++){
            bar[i] = '#';
        }
        percents = (100 * ((double)current / this->epochs));
        printf("\r\33[2K\rProgress: %s %d%c\n\033[A", bar, percents, 37);
    }
}

bool Ann::set_train_data(std::string filepath){
    std::ifstream readfile (filepath);
    if(readfile.is_open()){
        std::string line;
	    register uint32_t o = 0;
        while(getline(readfile, line)){
            uint32_t i = 0;
            std::vector<double> next_x;
            std::vector<double> next_y;
            while(line[i] != ' '){
            next_x.push_back(line[i] - '0');
                i++;
            }
            while(i < line.size()){
                i++;
                next_y.push_back(line[i] - '0');
            }
            this->x_train.push_back(next_x);
            this->y_train.push_back(next_y);
            this->training_order.push_back(o);
            o++;
        }
    }
    else{
	    return 1;
    }
    return 0;
}

bool Ann::set_train_data(std::vector<std::vector<double>> xtrain, std::vector<std::vector<double>> ytrain){
    for(register uint32_t i = 0; i < xtrain.size(); i++){
        this->x_train.push_back(xtrain[i]);
        this->training_order.push_back(i);
    }
    for(register uint32_t i = 0; i < ytrain.size(); i++){
        this->y_train.push_back(ytrain[i]);
    }

    return 0;
}

bool Ann::predict(std::vector<double>& input){

    if(input.size() != this->input.size()) return 1;
    set_input(input);
    feed_forward();
    for(register uint32_t i = 0; i < this->output.size(); i++){
        this->predicted[i] = this->output[i];
    }

    return 0;
}

void Ann::set_learning_rate(double learning_rate){
    this->learning_rate = learning_rate;
    return;
}

void Ann::set_epochs(uint32_t epochs){
    this->epochs = epochs;
    return;
}
void Ann::set_input(std::vector<double>& input){
    for(register uint32_t i = 0; i < this->input.size(); i++){
        this->input[i] = input[i];
    }
}

void Ann::feed_forward(){
    //print_network(1);
    for(register uint32_t i = 0; i < this->height; i++){
        double sum = this->hidden_bias[i];
        for(uint32_t j = 0; j < this->input.size(); j++){
            sum += this->input[j] * this->input_weights[j][i];
            //printf("input_w: %f\n", this->input_weights[j][i]);
        }
        //printf("\nSUM1: %f \n", sum);
        this->hidden[i] = (this->*activation)(sum);
        //printf("hidden[%d]: %f\n", i, this->hidden[i]);
    }
    /*printf("\n");
    printf("_____START_____\n");*/
    for(register uint32_t i = 0; i < (this->hidden.size() - this->height); i += this->height){
        for(register uint32_t j = 0; j < this->height; j++){
            double sum = this->hidden_bias[i + this->height + j];
            //printf("sum before: %f\n", sum);
            for(register uint32_t k = 0; k < this->height; k++){
                sum += this->hidden[i + k] * this->hidden_weights[i + k][j];
                /*printf("sum += h[%d]%f * h_w[%d][%d] = %f\n", (i + k), this->hidden[i + k], (i + k), j, this->hidden_weights[i + k][j]);
                printf("sum in loop: %f\n", sum);*/
            }
            this->hidden[i + this->height + j] = (this->*activation)(sum);
            /*printf("\nSUM2: %f \n", sum);
            printf("\nhidden[%d] = %f \n", (i + this->height + j), this->hidden[i + this->height + j]);
            print_hidden();*/
        }
        //print_network(1);
    }

    for(register uint32_t i = 0; i < this->output.size(); i++){
        double sum = this->output_bias[i];
        for(register uint32_t j = 0; j < this->height; j++){
            sum += this->hidden[(this->hidden.size() - this->height) + j] * output_weights[i][j];
            //printf("h[%lu] = %f, o_w[%d][%d] = %f\n", ((this->hidden.size() - this->height) + j), this->hidden[(this->hidden.size() - this->height) + j], i, j, output_weights[i][j]);
        }
        //printf("\nSUM3: %f \n", sum);
        this->output[i] = (this->*activation)(sum);
        /*printf("sum: %f\n", sum);
        printf("out[%d]: %f\n", i, this->output[i]);*/
    }
    return;
}

void Ann::back_propagate(std::vector<double>& expected){
    for(register uint32_t i = 0; i < this->output.size() && i < expected.size(); i++){
        const double error = expected[i] - output[i];
        this->output_error[i] = error * (this->*activation_d)(output[i]);
        /*printf("expected %f\n", expected[i]);
        printf("out_error %f\n", this->output_error[i]);*/
    }

    for(register uint32_t i = (this->hidden.size() - this->height); i < this->hidden.size(); i++){
        double error = 0;
        for(register uint32_t j = 0; j < this->output.size(); j++){
            error += this->output_error[j] * this->output_weights[j][i - (this->hidden.size() - this->height)];
        }
        this->hidden_error[i] = error * (this->*activation_d)(this->hidden[i]);
        /*printf("\nerror_last[%d]: %f \n", i, this->hidden_error[i]);
        print_hidden_error();*/
    }

    for(register uint32_t i = (this->hidden.size() - this->height); i >= this->height; i -= this->height){
        for(register uint32_t j = 0; j < this->height; j++){
            double error = 0;
            for(register uint32_t k = 0; k < this->height; k++){
                error += this->hidden_error[i + k] * this->hidden_weights[i - j - 1][k];
            }
            this->hidden_error[i - j - 1] = error * (this->*activation_d)(this->hidden[i - j - 1]);
            //printf("hidden[%d] = %f", (i - j - 1), this->hidden[i - j - 1]);
            //printf("\nhidden_error[%d]: %f \n", (i - j - 1), this->hidden_error[i - j - 1]);
        }
    }
    return;
}

void Ann::optimize(void){
    for(register uint32_t i = 0; i < this->input.size(); i++){
        for(register uint32_t j = 0; j < this->height; j++){
            const double change = this->hidden_error[j] * this->learning_rate;
            this->input_weights[i][j] += change * this->input[i];
            /*if(this->input_weights[i][j] > 10) this->input_weights[i][j] = 5;
            else if(this->input_weights[i][j] < -10) this->input_weights[i][j] = -5;*/
        }
    }
    //printf("START");
    for(register uint32_t i = 0; i < this->hidden.size(); i += this->height){
        for(register uint32_t j = 0; j < this->height; j++){
            const double change = this->hidden_error[i + j] * this->learning_rate;
            this->hidden_bias[i + j] += change;
            //printf("\nCHANGE: %f \n", change);
            /*printf("\nChanged: %f \n", this->hidden_bias[i + j]);
            printf("%f, %f, %f, %f\n", this->hidden_bias[0], this->hidden_bias[4], this->hidden_bias[8], this->hidden_bias[120]);
            printf("%f, %f, %f, %f\n", this->hidden_bias[1], this->hidden_bias[5], this->hidden_bias[9], this->hidden_bias[121]);
            printf("%f, %f, %f, %f\n", this->hidden_bias[2], this->hidden_bias[6], this->hidden_bias[10], this->hidden_bias[123]);
            printf("%f, %f, %f, %f\n", this->hidden_bias[3], this->hidden_bias[7], this->hidden_bias[11], this->hidden_bias[124]);*/
            if(i >= this->height){
                for(register uint32_t k = 0; k < this->height; k++){
                    this->hidden_weights[i - this->height + k][j] += change * this->hidden[i - this->height + k];
                    /*if(this->hidden_weights[i - this->height + k][j] > 10) this->hidden_weights[i - this->height + k][j] = 5;
                    else if(this->hidden_weights[i - this->height + k][j] < -10) this->hidden_weights[i - this->height + k][j] = -5;
                    /*printf(" ");
                    printf("h_w[%d][%d]\n", (i - this->height + k), j);*/
                }
            }
        }
    }
    for(register uint32_t i = 0; i < this->output.size(); i++){
        const double change = this->output_error[i] * this->learning_rate;
        this->output_bias[i] += change;
        //printf("Change %f\n", change);
        for(register uint32_t j = 0; j < this->height; j++){
            this->output_weights[i][j] += change * this->hidden[(this->hidden.size() - this->height) + j];
            /*if(this->output_weights[i][j] > 10) this->output_weights[i][j] = 5;
            else if(this->output_weights[i][j] < -10) this->output_weights[i][j] = -5;
            //printf("out_w[%d][%d] = %f\n", i, j, this->output_weights[i][j]);*/
        }
    }
    return;
}

void Ann::reset_hidden(void){
    for(uint32_t i = 0; i < this->hidden.size(); i++){
        this->hidden[i] = 0;
    }
}

double Ann::relu(double sum){
    //if(sum >= 20) return 15;
    if(sum >= 0) return sum;
    return sum * this->leak;
}

double Ann::derived_relu(double sum){
    if(sum > 0) return 1;
    return this->leak * sum;
}

double Ann::sigmoid(double sum){
    return (1 / (1 + pow(e, -sum)));
}

double Ann::derived_sigmoid(double sum){
    double num = this->sigmoid(sum);
    return (num * (1 - num));
}

double Ann::tanh(double sum){
    return (2 / (1 + pow(e, -(2*sum)))) - 1;
}

double Ann::derived_tanh(double sum){
    double num = this->tanh(sum);
    return 1 - pow(num, 2);
}

static double get_random(void){
    double ran = (1 - (2 * (rand()/(float)RAND_MAX)));
    return ran;
}

void Ann::setup(void){
    init_random();
    for(register uint32_t i = 0; i < this->input.size(); i++){
        this->input[i] = 0;
        for(register uint32_t j = 0; j < this->height; j++){
            this->input_weights[i][j] = get_random();
        }
    }
    for(register uint32_t i = 0; i < this->hidden.size(); i++){
        this->hidden[i] = 0;
        this->hidden_error[i] = 0;
        this->hidden_bias[i] = get_random();
        if(i < (this->hidden.size() - this->height)){
            for(register uint32_t j = 0; j < this->height; j++){
                this->hidden_weights[i][j] = get_random();
            }
        }
    }
    for(register uint32_t i = 0; i < this->output.size(); i++){
        this->output[i] = 0;
        this->output_error[i] = 0;
        this->output_bias[i] = get_random();
        for(register uint32_t j = 0; j < this->height; j++){
            this->output_weights[i][j] = get_random();
        }
    }
}

static void init_random(void){
    srand(time(NULL));
    return;
}

void Ann::shuffle(){
    for(register uint32_t i = 0; i < this->x_train.size(); i++){
        const uint32_t r = rand() % x_train.size();
        const uint32_t temp = this->training_order[i];
        this->training_order[i] = this->training_order[r];
        this->training_order[r] = temp;

    }
    return;
}

void Ann::save(std::ofstream& of){
    
    
}
void Ann::load(std::ifstream& inf){

}