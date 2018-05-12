#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "factor.h"
#include "predict.h"

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
    std::size_t size() const { return m_data.size(); }
    void readNextRow(std::istream& str) {
        std::string line;
        std::getline(str, line);

        std::stringstream lineStream(line);
        std::string cell;

        m_data.clear();
        while (std::getline(lineStream, cell, sep)) {
            m_data.push_back(cell);
        }
        // This checks for a trailing comma with no data after it.
        if (!lineStream && cell.empty()) {
            // If there was a trailing comma then add an empty element.
            m_data.push_back("");
        }
    }

    explicit CSVRow(char sep) : sep(sep) {}

   private:
    char sep;
    std::vector<std::string> m_data;
};

std::istream& operator>>(std::istream& str, CSVRow& data) {
    data.readNextRow(str);
    return str;
}

void InsertRow(std::vector<std::vector<std::optional<double>>>& dataset,
               const CSVRow& row, const std::set<size_t>& dropped_columns) {
    dataset.emplace_back();
    for (size_t i = 0; i < row.size(); ++i) {
        if (dropped_columns.find(i) == dropped_columns.end()) {
            std::string value = row[i];
            std::replace(value.begin(), value.end(), ',', '.');
            if (value != "") {
                dataset.back().push_back(std::stof(value));
            } else {
                dataset.back().push_back(std::nullopt);
            }
        }
    }
}

std::tuple<MatrixXd, MatrixXb> ToEigenMatrices(
    const std::vector<std::vector<std::optional<double>>> data,
    size_t timestamp_row) {
    int T = data.size();
    int n = data[0].size();

    MatrixXd matrix = MatrixXd::Zero(n - 1, T);
    MatrixXb omega = MatrixXb::Zero(n - 1, T);

    for (int i = 0; i < n; ++i) {
        if (i == timestamp_row) {
            continue;
        }
        for (int j = 0; j < T; ++j) {
            if (data[j][i]) {
                matrix(i - (i > timestamp_row), j) = *data[j][i];
                omega(i - (i > timestamp_row), j) = true;
            } else {
                matrix(i - (i > timestamp_row), j) = 0;
                omega(i - (i > timestamp_row), j) = false;
            }
        }
    }
    return std::make_tuple(matrix, omega);
}

int main(int argc, const char* argv[]) {
    po::options_description desc("Allowed options");
    std::vector<int> default_lags = {1, 5, 10};
    std::vector<size_t> default_drop_columns = {1};

    desc.add_options()
        ("help", "produce help message")
        ("dataset_path", po::value<std::string>(), "path to dataset")
        ("timestamp_column", po::value<size_t>()->default_value(0), "timestamp column")
        ("steps", po::value<size_t>()->default_value(100), "optimization steps")
        ("train_start", po::value<long>(), "train start timestamp")
        ("train_end", po::value<long>(), "train end timestamp")
        ("test_start", po::value<long>()->default_value(-1), "test start timestamp")
        ("test_end", po::value<long>()->default_value(-1), "test end timestamp")
        ("lat_dim", po::value<size_t>()->default_value(2), "latent embedding dimension")
        ("drop_columns", po::value<std::vector<size_t>>()->multitoken()->default_value(default_drop_columns, ""), "drop columns list")
        ("lags", po::value<std::vector<int>>()->multitoken()->default_value(default_lags, "1 5 10"), "lags list")
        ("verbose", po::value<bool>()->default_value(false), "verbose all shit")
        ("separator", po::value<char>()->default_value(';'), "separator for csv")
        ("predictions_out", po::value<string>()->default_value(""), "predictions output filename")
        ("lambdaX", po::value<double>()->default_value(1), "lambdaX")
        ("lambdaW", po::value<double>()->default_value(1), "lambdaW")
        ("lambdaF", po::value<double>()->default_value(1), "lambdaF")
        ("nu", po::value<double>()->default_value(1), "nu");

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
    auto verbose = vm["verbose"].as<bool>();
    auto lat_dim = vm["lat_dim"].as<size_t>();
    auto lambdaX = vm["lambdaX"].as<double>();
    auto lambdaW = vm["lambdaW"].as<double>();
    auto lambdaF = vm["lambdaF"].as<double>();
    auto nu = vm["nu"].as<double>();

    std::set<size_t> dropped_columns;
    for (const auto& d : drop_columns) {
        dropped_columns.insert(d);
    }

    std::ifstream file(dataset_path);
    CSVRow row(sep);
    file >> row;
    size_t n = row.size() - drop_columns.size();

    std::vector<std::vector<std::optional<double>>> train_data;
    std::vector<std::vector<std::optional<double>>> test_data;

    while (file >> row) {
        if (std::stol(row[timestamp_column]) >= train_start &&
            std::stol(row[timestamp_column]) < train_end) {
            InsertRow(train_data, row, dropped_columns);
        }
        if (std::stol(row[timestamp_column]) >= test_start &&
            std::stol(row[timestamp_column]) < test_end) {
            InsertRow(test_data, row, dropped_columns);
        }
    }

    int timestamp_shitf = 0;
    for (auto i : dropped_columns) {
        if (i < timestamp_column) {
            ++timestamp_shitf;
        }
    }
    timestamp_column -= timestamp_shitf;

    double time_step = (*train_data.back()[timestamp_column] -
                        *train_data[0][timestamp_column]) /
                       (train_data.size() - 1);

    auto [train_matrix, train_omega] =
        ToEigenMatrices(train_data, timestamp_column);

    MatrixXd test_matrix;
    MatrixXb test_omega;
    if (test_data.size() > 0) {
        std::tie(test_matrix, test_omega) =
            ToEigenMatrices(test_data, timestamp_column);
    }

    auto factor = Factorize(train_matrix, train_omega,
                            {lags, lambdaW, lambdaX, lambdaF, nu},
                            lat_dim, steps, verbose);
    if (verbose) {
        std::cout << factor.F << "\n";
        std::cout << factor.W << "\n";
        std::cout << factor.X << "\n";
    }
    size_t test_start_index = (test_start - train_start) / time_step;
    size_t test_end_index = test_end / time_step;
    if (test_data.size() > 0) {
        test_end_index = test_start_index + test_data.size();
    }
    auto predictions = Predict(factor, lags, test_start_index, test_end_index);
    std::cout << "RMSE: " << RMSE(test_matrix, predictions, test_omega) << "\n";
    std::cout << "ND: " << ND(test_matrix, predictions, test_omega) << "\n";
}
