#include <iostream>
#include <fstream>
#include <random>
#include <vector>
using namespace std;

void init(vector<double> &X, int n) {
	/* Initialize the input vector */
	default_random_engine generator;
	uniform_real_distribution<double> unif(0, 1);

	for (int i = 0; i < n; ++i)
		X.push_back(unif(generator) <= 0.5 ? -1 : 1);
}

void mcmc(vector<vector<double> > W, vector<double> Y) {
	/* Run the MCMC algorithm to determine the input vector */
	vector<double> X;
	int m = W.size(), n = W[0].size();

	/* Initialize the input vector */
	init(X, n);
}

int main(int argc, char **argv) {
	int m, n;
	double x;
	vector<double> Y;
	vector<vector<double> > W;
	ifstream f(argv[1]);

	/* Read the features matrix */
	f >> m >> n;
	for (int i = 0; i < m; ++i) {
		vector<double> row;
		for (int j = 0; j < n; ++j) {
			f >> x;
			row.push_back(x);
		}
		W.push_back(row);
	}

	/* Read the observations vector */
	for (int i = 0; i < m; ++i) {
		f >> x;
		Y.push_back(x);
	}

	mcmc(W, Y);

	return 0;
}