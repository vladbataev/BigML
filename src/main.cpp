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
                size_t idx;
                dataset.back().push_back(std::stof(value, &idx));
                if (idx != value.size()) {
                    throw std::runtime_error("invalid value in csv row");
                }
            } else {
                dataset.back().push_back(std::nullopt);
            }
        }
    }
}

std::vector<std::tuple<double, double>> Standardize(MatrixXd& matrix) {
    int T = matrix.cols();
    int n = matrix.rows();
    std::vector<std::tuple<double, double>> statistics;
    for (int i = 0; i < n; ++i) {
        double mean = matrix.row(i).mean();
        double sigma = sqrt((matrix.row(i).transpose() - mean * VectorXd::Ones(T)).squaredNorm() / T);
        matrix.row(i) = ((matrix.row(i).transpose() - mean * VectorXd::Ones(T)) / sigma).transpose();
        statistics.emplace_back(mean, sigma);
    }
    return statistics;
};

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

void PrintHeader(const std::vector<std::string>& header, std::ostream& out) {
    if (!header.empty()) {
        for (size_t i = 0; i < header.size(); i++) {
            if (i != 0) {
                out << ",";
            }
            out << header[i];
        }
        out << "\n";
    }
}

void SaveToCSV(const MatrixXd& matrix, std::string filename, const std::vector<std::string>& header) {
    std::ofstream myfile;
    myfile.open(filename);
    myfile.precision(10);
    if (myfile.fail()) {
        throw std::runtime_error("failed to open " + filename);
    }
    PrintHeader(header, myfile);
    for (size_t i = 0; i < matrix.cols(); i++) {
        if (i != 0) {
            myfile << "\n";
        }
        for (size_t j = 0; j < matrix.rows(); j++) {
            if (j != 0) {
                myfile << ",";
            }
            myfile << matrix(j, i);
        }
    }
    myfile.close();
}

void SaveWithTimestamps(const MatrixXd& predictions,
                     std::vector<double> timestamps,
                     std::string out_csv,
                     std::vector<std::string> header) {
    std::ofstream myfile;
    myfile.open(out_csv);
    if (myfile.fail()) {
        throw std::runtime_error("failed to open " + out_csv);
    }
    myfile.precision(10);
    PrintHeader(header, myfile);
    for (size_t i = 0; i < predictions.cols(); ++i) {
        myfile << timestamps[i];
        for (size_t j = 0; j < predictions.rows(); ++j) {
            myfile << "," << predictions(j , i);
        }
        myfile << "\n";
    }
    myfile.close();
}


int main(int argc, const char* argv[]) {
    po::options_description desc("Allowed options");
    std::vector<int> default_lags = {1, 5, 10, 20, 25, 100};
    std::vector<size_t> default_drop_columns = {};

    desc.add_options()
        ("help", "produce help message")
        ("dataset_path", po::value<std::string>(), "path to dataset")
        ("timestamp_column", po::value<size_t>()->default_value(0), "timestamp column")
        ("steps", po::value<size_t>()->default_value(100), "optimization steps")
        ("train_start", po::value<long>(), "train start timestamp")
        ("train_end", po::value<long>(), "train end timestamp")
        ("test_start", po::value<long>(), "test start timestamp")
        ("test_end", po::value<long>(), "test end timestamp")
        ("lat_dim", po::value<size_t>()->default_value(2), "latent embedding dimension")
        ("drop_columns",
           po::value<std::vector<size_t>>()->multitoken()->default_value(default_drop_columns, ""),
            "drop columns list, default is empty")
        ("lags",
            po::value<std::vector<int>>()->multitoken()->default_value(default_lags, "1 5 10 20 25 100"),
            "lags list")
        ("eval", po::value<bool>()->default_value(false), "calculate metrics if known true values")
        ("standardize", po::value<bool>()->default_value(true), "standardize train data")
        ("verbose", po::value<bool>()->default_value(false), "verbose all shit")
        ("separator", po::value<char>()->default_value(';'), "separator for csv")
        ("predictions_out",
           po::value<std::string>()->default_value("predictions.csv"),
           "predictions output filename")
        ("factor_out",
           po::value<std::string>()->default_value(""),
           "write result matrices to files with specified prefix")
        ("lambdaX", po::value<double>()->default_value(1), "lambdaX")
        ("lambdaW", po::value<double>()->default_value(1), "lambdaW")
        ("lambdaF", po::value<double>()->default_value(1), "lambdaF")
        ("eta", po::value<double>()->default_value(1), "eta")
        ("logs_file", po::value<std::string>()->default_value(""), "if non empty than store logs to this file");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help") or argc == 1) {
        std::cout << desc << "\n";
        return 1;
    }

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
    auto standardize = vm["standardize"].as<bool>();
    auto lat_dim = vm["lat_dim"].as<size_t>();
    auto lambdaX = vm["lambdaX"].as<double>();
    auto lambdaW = vm["lambdaW"].as<double>();
    auto lambdaF = vm["lambdaF"].as<double>();
    auto eta = vm["eta"].as<double>();
    auto eval =  vm["eval"].as<bool>();
    auto predictions_out = vm["predictions_out"].as<std::string>();
    auto factor_out = vm["factor_out"].as<std::string>();
    auto logs_file_name = vm["logs_file"].as<std::string>();

    std::ofstream logs_file;
    bool logs_enabled;
    if (logs_file_name.size() > 0) {
        logs_file.open(logs_file_name);
        logs_enabled = true;
    } else {
        logs_enabled = false;
    }

    std::set<size_t> dropped_columns;
    for (const auto& d : drop_columns) {
        dropped_columns.insert(d);
    }

    auto expect = [](bool condition, std::string message) {
        if (!condition) {
            throw std::invalid_argument(message);
        }
    };

    expect(!(lambdaF < 0 || lambdaW < 0 || lambdaX < 0 || eta < 0), "non-convex optimization target");

    expect(train_start < train_end <= test_start < test_end, "Timestamps order: train_start < train_end <= test_start < test_end");

    std::ifstream file(dataset_path);
    expect(!file.fail(), "failed to open " + dataset_path);

    CSVRow row(sep);
    file >> row;
    std::vector<std::string> output_header = {row[timestamp_column]}; //timestamp column is always first
    for (size_t i = 0; i < row.size(); i++) {
        if (dropped_columns.find(i) == dropped_columns.end() &&
                i != timestamp_column) {
            output_header.push_back(row[i]);
        }
    }

    size_t n = row.size() - drop_columns.size();
    size_t first_size = row.size();

    std::vector<std::vector<std::optional<double>>> train_data;
    std::vector<std::vector<std::optional<double>>> test_data;

    std::vector<double> train_timestamps;
    std::vector<double> test_timestamps;
    while (file >> row) {
        expect(row.size() == first_size, "invalid csv file");
        if (std::stol(row[timestamp_column]) >= train_start &&
            std::stol(row[timestamp_column]) < train_end) {
            InsertRow(train_data, row, dropped_columns);
            train_timestamps.push_back(std::stof(row[timestamp_column]));
        }
        if (std::stol(row[timestamp_column]) >= test_start &&
            std::stol(row[timestamp_column]) < test_end) {
            InsertRow(test_data, row, dropped_columns);
            test_timestamps.push_back(std::stof(row[timestamp_column]));
        }
    }

    expect(!train_data.empty(), "Train_data is empty. Double check parameters.");
    expect(!eval || !test_data.empty(), "Test data is empty. Nothing to eval.");

    int timestamp_shift = 0;
    for (auto i : dropped_columns) {
        if (i < timestamp_column) {
            ++timestamp_shift;
        }
    }
    timestamp_column -= timestamp_shift;

    double time_step = (*train_data.back()[timestamp_column] -
                        *train_data[0][timestamp_column]) /
                       (train_data.size() - 1);

    auto [train_matrix, train_omega] =
        ToEigenMatrices(train_data, timestamp_column);

    std::vector<std::tuple<double, double>> statistics;
    if (standardize) {
        statistics = Standardize(train_matrix);
    }
    MatrixXd test_matrix;
    MatrixXb test_omega;
    if (test_data.size() > 0) {
        std::tie(test_matrix, test_omega) =
            ToEigenMatrices(test_data, timestamp_column);
    }

    auto factor = Factorize(train_matrix, train_omega,
                            {lags, lambdaW, lambdaX, lambdaF, eta},
                            lat_dim, steps, logs_file, logs_enabled, verbose);

    if (!factor_out.empty()) {
        SaveWithTimestamps(factor.X, train_timestamps, factor_out + "_X.csv", {});
        SaveToCSV(factor.W, factor_out + "_W.csv", {});
        SaveToCSV(factor.F.transpose(), factor_out + "_F.csv", {output_header.begin() + 1, output_header.end()});
    }
    size_t test_start_index = ceil((test_start - train_start) / time_step);
    size_t test_end_index = ceil((test_end - train_start) / time_step);
    if (test_data.size() > 0) {
        test_end_index = test_start_index + test_data.size();
    }

    auto predictions = Predict(factor, lags, test_start_index, test_end_index);
    if (standardize) {
        for (int i = 0; i < predictions.rows(); ++i) {
            predictions.row(i) = predictions.row(i) * std::get<1>(statistics[i]) +
                    VectorXd::Ones(predictions.cols()).transpose() * std::get<0>(statistics[i]);
        }
    }

    if (eval) {
        SaveWithTimestamps(predictions, test_timestamps, predictions_out, output_header);
        std::cout << "RMSE: " << RMSE(test_matrix, predictions, test_omega) << "\n";
        std::cout << "ND: " << ND(test_matrix, predictions, test_omega) << "\n";
        if (logs_file_name.size() > 0) {
            logs_file << "RMSE: " << RMSE(test_matrix, predictions, test_omega) << "\n";
            logs_file << "ND: " << ND(test_matrix, predictions, test_omega) << "\n";
        }
    } else {
        test_timestamps.clear();
        for (int i = 0; i < predictions.cols(); i++) {
            test_timestamps.push_back(test_start + time_step * i);
        }
        SaveWithTimestamps(predictions, test_timestamps, predictions_out, output_header);
    }
}
