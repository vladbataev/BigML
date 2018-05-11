#include "factor.h"
#include "predict.h"
#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <set>

#include <Eigen/Dense>
#include <boost/program_options.hpp>


using namespace std;
using namespace Eigen;

namespace po = boost::program_options;

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

    explicit CSVRow(char sep): sep(sep)
    {}

private:
    char sep;
    std::vector<std::string>    m_data;
};

std::istream& operator>>(std::istream& str, CSVRow& data) {
    data.readNextRow(str);
    return str;
}


void insert_row(std::vector<std::vector<float>>& dataset,  const CSVRow& row,
                const std::set<size_t>& dropped_columns, size_t timestamp_column, long start_t, long end_t) {
    dataset.push_back(std::vector<float>());
    if (std::stol(row[timestamp_column]) >= start_t && std::stol(row[timestamp_column]) <= end_t) {
        for (size_t i = 0; i < row.size(); ++i) {
            if (dropped_columns.find(i) == dropped_columns.end()) {
                std::string value = row[i];
                std::replace(value.begin(), value.end(), ',', '.');
                dataset.back().push_back(std::stof(value));
            }
        }
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
    std::vector<int> default_lags = {1, 5, 10};
    std::vector<size_t> default_drop_columns = {};

    desc.add_options()
            ("help", "produce help message")
            ("dataset_path", po::value<std::string>(),  "path to dataset")
            ("timestamp_column", po::value<size_t >()->default_value(0), "timestamp column")
            ("train_start", po::value<long>(), "train start timestamp")
            ("train_end", po::value<long>(), "train end timestamp")
            ("test_start", po::value<long>()->default_value(-1), "test start timestamp")
            ("test_end", po::value<long>()->default_value(-1), "test end timestamp")
            ("drop_columns", po::value<std::vector<size_t> >()->multitoken()->default_value(default_drop_columns, ""),
                 "drop columns list")
            ("lags", po::value<std::vector<int> >()->multitoken()->default_value(default_lags, "1 5 10"),
                 "lags list")
            ("separator", po::value<char>()->default_value(';'),  "separator for csv")
            ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    auto dataset_path = vm["dataset_path"].as<std::string>();
    auto train_start = vm["train_start"].as<long>();
    auto train_end = vm["train_end"].as<long>();
    auto test_start = vm["test_start"].as<long>();
    auto test_end = vm["test_end"].as<long>();
    auto sep = vm["separator"].as<char>();
    auto lags = vm["lags"].as<std::vector<int>>();
    auto drop_columns = vm["drop_columns"].as<std::vector<size_t>>();
    auto timestamp_column = vm["timestamp_column"].as<size_t>();

    std::set<size_t> dropped_columns;
    for (const auto& d: drop_columns) {
        dropped_columns.insert(d);
    }
    dropped_columns.insert(timestamp_column);

    std::ifstream file(dataset_path);
    CSVRow row(sep);
    file >> row;
    std::vector<std::vector<float>> train_data;
    std::vector<std::vector<float>> test_data;
    while(file >> row) {
        insert_row(train_data, row, dropped_columns, timestamp_column, train_start, train_end);
        insert_row(test_data, row, dropped_columns, timestamp_column, test_start, test_end);
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

    train_matrix = MatrixXd::Random(50, 50);

    size_t lat_dim = 2;
    size_t steps = 20;
    auto factor = Factorize(train_matrix, Regularizer{lags, 1.0, 1.0, 1.0}, lat_dim, steps);
    std::cout << factor.F << "\n";
    std::cout << factor.W << "\n";
    std::cout << factor.X << "\n";

    std::cout << Predict(factor, lags);

}
