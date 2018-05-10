#include "W.h"
#include "X.h"
#include "F_optimize.h"

#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include <Eigen/Dense>
#include <boost/program_options.hpp>

using namespace std;
using namespace Eigen;

namespace po = boost::program_options;

struct Regularizer {
    vector<int> lags;
    double lambdaW;
};

struct Factorization {
};

Factorization Factorize(MatrixXd X, Regularizer opts, size_t steps) {
    CachedWTransform Wt(opts.lags);
    MatrixXd W(X.rows(), opts.lags.size());
    MatrixXd F;

    for (size_t i = 0; i < steps; i++) {
    }
}

class CSVRow {
public:
    std::string const& operator[](std::size_t index) const {
        return m_data[index];
    }
    std::size_t size() const {
        return m_data.size();
    }
    void readNextRow(std::istream& str) {
        std::string         line;
        std::getline(str, line);

        std::stringstream   lineStream(line);
        std::string         cell;

        m_data.clear();
        while(std::getline(lineStream, cell, ';')) {
            m_data.push_back(cell);
        }
        // This checks for a trailing comma with no data after it.
        if (!lineStream && cell.empty()) {
            // If there was a trailing comma then add an empty element.
            m_data.push_back("");
        }
    }
private:
    std::vector<std::string>    m_data;
};

std::istream& operator>>(std::istream& str, CSVRow& data) {
    data.readNextRow(str);
    return str;
}


void insert_row(std::vector<std::vector<float>>& dataset,  const CSVRow& row,
                const std::vector<int>& data_columns) {
    dataset.push_back(std::vector<float>());
    for (const auto& column: data_columns) {
        std::string value = row[column];
        std::replace(value.begin(), value.end(), ',', '.');
        dataset.back().push_back(std::stof(value));
    }
}

void to_eigen_matrix(const std::vector<std::vector<float>> data, MatrixXd& matrix) {
    int T = data.size();
    int n = data[0].size();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < T; ++j) {
            matrix(i, j) = data[j][i];
        }
    }
}

int main(int argc, const char* argv[]) {
    po::options_description desc("Allowed options");
    desc.add_options()
            ("help", "produce help message")
            ("dataset_path", po::value<std::string>(),  "path to dataset")
            ("train_start", po::value<long>(), "train start timestamp")
            ("train_end", po::value<long>(), "train end timestamp")
            ("test_start", po::value<long>()->default_value(-1), "test start timestamp")
            ("test_end", po::value<long>()->default_value(-1), "test end timestamp")
            ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    auto dataset_path = vm["dataset_path"].as<std::string>();
    auto train_start = vm["train_start"].as<long>();
    auto train_end = vm["train_end"].as<long>();
    auto test_start = vm["test_start"].as<long>();
    auto test_end = vm["test_end"].as<long>();

    std::ifstream file(dataset_path);
    CSVRow row;
    file >> row;
    std::vector<int> data_columns = {2, 3, 4, 5};
    std::vector<std::vector<float>> train_data;
    std::vector<std::vector<float>> test_data;
    while(file >> row) {
        long timestamp = std::stol(row[0]);
        if (train_start <= timestamp <= train_end) {
            insert_row(train_data, row, data_columns);
        }
        if (test_start <= timestamp <= test_end) {
            insert_row(test_data, row, data_columns);
        }
    }

    int train_T = train_data.size();
    int train_N = train_data[0].size();
    MatrixXd train_matrix = MatrixXd::Zero(train_N, train_T);
    to_eigen_matrix(train_data, train_matrix);

    if (test_data.size() > 0) {
        int test_T = test_data.size();
        int test_N = test_data[0].size();
        MatrixXd test_matrix = MatrixXd::Zero(train_N, train_T);
        to_eigen_matrix(test_data, test_matrix);
    }
    std::cout << train_matrix;
}
