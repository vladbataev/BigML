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
        while(std::getline(lineStream, cell, sep)) {
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


void insert_row(std::vector<std::vector<std::optional<double>>>& dataset,  const CSVRow& row,
                const std::set<size_t>& dropped_columns) {
    dataset.emplace_back();
    for (size_t i = 0; i < row.size(); ++i) {
        if (dropped_columns.find(i) == dropped_columns.end()) {
            std::string value = row[i];
            std::replace(value.begin(), value.end(), ',', '.');
            if (value != "" ) {
                dataset.back().push_back(std::stof(value));
            } else {
                dataset.back().push_back(std::nullopt);
            }
        }
    }
}

void to_eigen_matrix(const std::vector<std::vector<std::optional<double>>> data, MatrixXd& matrix, MatrixXb& Omega) {
    int T = data.size();
    int n = data[0].size();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < T; ++j) {
            if (data[j][i]) {
                matrix(i, j) = *data[j][i];
                Omega(i, j) = true;
            } else {
                matrix(i, j) = 0;
                Omega(i, j) = false;
            }
        }
    }
}

int main(int argc, const char* argv[]) {
    po::options_description desc("Allowed options");
    std::vector<int> default_lags = {1, 5, 10};
    std::vector<size_t> default_drop_columns = {0, 1};

    desc.add_options()
            ("help", "produce help message")
            ("dataset_path", po::value<std::string>(),  "path to dataset")
            ("timestamp_column", po::value<size_t >()->default_value(0), "timestamp column")
            ("steps", po::value<size_t >()->default_value(100), "optimization steps")
            ("train_start", po::value<long>(), "train start timestamp")
            ("train_end", po::value<long>(), "train end timestamp")
            ("test_start", po::value<long>()->default_value(-1), "test start timestamp")
            ("test_end", po::value<long>()->default_value(-1), "test end timestamp")
            ("lat_dim", po::value<size_t>()->default_value(2), "latent embedding dimension")
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
    auto steps = vm["steps"].as<size_t>();
    auto lat_dim = vm["lat_dim"].as<size_t>();

    std::set<size_t> dropped_columns;
    for (const auto& d: drop_columns) {
        dropped_columns.insert(d);
    }
    dropped_columns.insert(timestamp_column);

    std::ifstream file(dataset_path);
    CSVRow row(sep);
    file >> row;
    size_t n = row.size() - drop_columns.size();

    std::vector<std::vector<std::optional<double>>> train_data;
    std::vector<std::vector<std::optional<double>>> test_data;
    while(file >> row) {
        if (std::stol(row[timestamp_column]) >= train_start && std::stol(row[timestamp_column]) <= train_end) {
            insert_row(train_data, row, dropped_columns);
        }
        if (std::stol(row[timestamp_column]) >= test_start && std::stol(row[timestamp_column]) <= test_end) {
            insert_row(test_data, row, dropped_columns);
        }
    }

    int train_T = train_data.size();
    int train_N = train_data[0].size();
    MatrixXd train_matrix = MatrixXd::Zero(train_N, train_T);
    MatrixXb train_omega = MatrixXb::Zero(train_N, train_T);
    to_eigen_matrix(train_data, train_matrix, train_omega);

    if (test_data.size() > 0) {
        int test_T = test_data.size();
        int test_N = test_data[0].size();
        MatrixXd test_matrix = MatrixXd::Zero(train_N, train_T);
        MatrixXb test_omega = MatrixXb::Zero(test_N, test_T);
        to_eigen_matrix(test_data, test_matrix, test_omega);
    }

    size_t lat_dim = 2;
    auto factor = Factorize(train_matrix,
            train_omega,
            Regularizer{lags, 1.0, 1.0, 1.0},
            lat_dim,
            steps,
            true);
    std::cout << factor.F << "\n";
    std::cout << factor.W << "\n";
    std::cout << factor.X << "\n";

    std::cout << Predict(factor, lags);

}
