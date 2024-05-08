#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <map>
#include <algorithm>

using namespace std;

struct Data
{
    vector<double> features;
    bool label;
};

//Reads data from a csv file as training data
void loadTrainingData(const string& filename, vector<Data> &data) 
{
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Unable to open file " << filename << endl;
        return;
    }

    string line;
    while(getline(file,line))
    {
        stringstream ss(line);
        string token;
        Data inputData;

        while(getline(ss, token, ','))
        {
            inputData.features.push_back(stod(token));
        }

        inputData.label = static_cast<int>(inputData.features.back());
        inputData.features.pop_back();

        data.push_back(inputData);
    }
}

//Simple function to train a Naive Bayes Classifier
void train(const vector<Data> &data, 
map<int, map<int, map<double, int>>> &featureCount,
map<int, int> &classCount, map<int, int> &totalFeatures)
{
    //Count number of features
    int num_features = data[0].features.size();

    //Goes through every Data item in data
    for (const Data &e : data)
    {
        //Counts every label and increments them accordingly
        int label = e.label;
        classCount[label]++;

        //Counts occurence f each feature given label
        for (int i = 0; i < num_features; i++)
        {
            double feature = e.features[i];

            featureCount[label][i][feature]++;
            totalFeatures[label]++;
        }

    }
}

//Function to classify an email using the trained classifier
bool classify(const Data &data, 
map<int, map<int, map<double, int>>> &featureCount,
map<int, int> &classCount, map<int, int> &totalFeatures,
int numFeatures, int totEmail)
{
    map<int, double> probability;

    //Calculate probability of each class
    for (const auto &c : classCount)
    {
        int label = c.first;
        int value = c.second;

        double cProb = log(static_cast<double>(value)/totEmail);

        for(int i = 0; i < numFeatures; i++)
        {
            double featureVal = data.features[i];

            if (featureCount.at(label).at(i).find(featureVal) != featureCount.at(label).at(i).end())
            {
                int count = featureCount.at(label).at(i).at(featureVal);
                double prob = static_cast<double>(count)/totalFeatures.at(label);
                cProb += log(prob);
            }
            else
            {
                cProb += log(0.001);
            }
        }

        probability[label] = cProb;
    }

    //Determining the class with highest probability
    int bestLabel = -1;
    double maxProb = -9999999;

    for (const auto &p : probability)
    {
        if (p.second > maxProb)
        {
            maxProb = p.second;
            bestLabel = p.first;
        }
    }

    return bestLabel;
}

int main()
{
    vector<Data> trainingData;
    loadTrainingData("spambase.data", trainingData);

    map<int, map<int, map<double, int>>> featureCount;
    map<int, int> classCount;
    map<int, int> totalFeatures;

    //Training
    train(trainingData, featureCount, classCount, totalFeatures);

    //Evaluating
    int correct = 0;
    int emailCount = trainingData.size();
    int numFeatures = trainingData[0].features.size();

    //Classifying each email
    for (const auto & m : trainingData)
    {
        bool label = classify(m, featureCount, classCount, totalFeatures, numFeatures, emailCount);
        bool actual = m.label;

        if (label == actual)
            correct++;
    }

    double accuracy = ((double)correct/emailCount) * 100;
    cout << "Total Correct: " << correct << endl;
    cout << "Accuracy: " << accuracy << "%" << endl;
    
    /*
    //Code snippit to verifying the data
    cout << "Total amount of emails: " << trainingData.size() << endl;
    cout << "Total amount of spams: " << classCount[1] << endl;
    cout << "Total amount of hams: " << classCount[0] << endl;

    int yes, no;
    double sum;
    sum = yes = no = 0;

    for(int i = 0; i < trainingData.size(); i++)
    {
        sum += trainingData[i].features.size();

        if (trainingData[i].label)
            yes++;
        else if (!trainingData[i].label)
            no++;
    }
    
    cout << "Total amount of emails: " << yes + no << endl;
    cout << "Total amount of spams: " << yes << endl;
    cout << "Total amount of hams: " << no << endl;
    cout << "Average amount of featrures: " << sum/(yes+no) << endl;
    */

    return 0;
}