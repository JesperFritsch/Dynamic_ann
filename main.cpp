#include "header.h"

std::vector<std::vector<double>> pred =  {
    {0,0,1,
    0,0,1,
    0,0,1},

    {1,0,0,
    1,0,0,
    1,0,0},

    {0,1,0,
    0,1,0,
    0,0,1},

    {1,0,1,
    0,1,0,
    0,0,1},

    {1,0,0,
    0,1,0,
    0,0,1},

    {1,0,0,
    0,1,0,
    0,0,0},

    {1,1,1,
    1,1,1,
    1,1,1},

    {1,0,0,
    1,1,0,
    1,0,0},

    {0,0,0,
    1,1,1,
    0,0,0},

    {0,1,0,
    0,1,0,
    0,1,0},

    {0,1,0,
    0,0,0,
    1,0,1},

    {0,1,0,
    0,0,0,
    1,0,0},

    {0,1,0,
    0,1,0,
    1,0,1},

    {1,1,0,
    1,1,0,
    0,0,0},

    {1,0,1,
    1,1,0,
    0,1,0},
};

int main(void){

    std::string filepath = "training_data3.txt";

    Ann model(9, 10, 30, 1, 0, "tanh");
    model.set_epochs(300);
    model.set_learning_rate(0.001);
    model.set_train_data(filepath);
    model.train();

    /*for(uint8_t i = 0; i < pred.size(); i++){
        model.predict(pred[i]);
        printf("%f\n", model.predicted[0]);
    
    }*/

    for(uint8_t i = 0; i < pred.size(); i++){
        model.predict(pred[i]);
        printf("\n%.0f| %.0f| %.0f\n", pred[i][0], pred[i][1], pred[i][2]);
        printf("%.0f| %.0f| %.0f\n", pred[i][3], pred[i][4], pred[i][5]);
        printf("%.0f| %.0f| %.0f\n", pred[i][6], pred[i][7], pred[i][8]);
        printf("\n");
        printf("\nPredicted: %.1f\n", model.predicted[0]);
        //printf("\nTrue: %.1f | False %.1f\n", model.predicted[0], model.predicted[1]);
        //model.print_network(1);
        printf("________________________\n");
    }
    

    return 0;
}