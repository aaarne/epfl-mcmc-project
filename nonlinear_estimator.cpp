#include <iostream>
#include <fstream>
#include <math.h>
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

double relu(double x) {
	return x < 0 ? 0 : x;
}

vector<double> matmul(vector<vector<double> > W, vector<double> X) {
	vector<double> res;
	int m = W.size(), n = W[0].size();
	int aux;

	for (int i = 0; i < m; ++i) {
		aux = 0;
		for (int j = 0; j < n; ++j)
			aux += W[i][j] * X[j];
		res.push_back(aux);
	}

	return res;
}

double energy(vector<vector<double> > W, vector<double> Y, vector<double> X) {
	double en = 0;
	int m = W.size(), n = W[0].size();
	vector<double> aux;

	/* Multiply W and X */
	aux = matmul(W, X);

	/* Compute the energy */
	for (int i = 0; i < m; ++i)
		en += pow(Y[i] - relu(aux[i] / sqrt(n)), 2);

	return en;
}

vector<double> transition(vector<double> X, default_random_engine generator) {
	int n = X.size();
	uniform_real_distribution<double> unif(0, n);
	int ind = floor(unif(generator));

	/* Flip the chosen state */
	X[ind] = -X[ind];

	return X;
}

void mcmc(vector<vector<double> > W, vector<double> Y) {
	/* Run the MCMC algorithm to determine the input vector */
	vector<double> X, aux_X;
	int m = W.size(), n = W[0].size();
	int max_steps = 1000, step = 0;
	double beta = 2;
	double energ, aux_energ;
	double accept_prob, prob;
	default_random_engine generator;
	uniform_real_distribution<double> unif(0, 1);

	/* Initialize the input vector */
	init(X, n);
	energ = energy(W, Y, X);
	cout << "Initial energy " << energ << "\n";

	/* Minimize the energy */
	while (true) {
		/* Compute a transition */
		aux_X = transition(X, generator);
		aux_energ = energy(W, Y, aux_X);

		/* Compute the acceptance probabiility */
		accept_prob = min(1.0, exp(-beta * (aux_energ - energ)));

		/* Decide whether to do the transition or not */
		prob = unif(generator);
		if (prob <= accept_prob) {
			X = aux_X;
			energ = aux_energ;
			++step;
			cout << "Step " << step << " energy " << energ << "\n";
			/* End the chain if we reached a lower enough energy */
			if (step == max_steps)
				break;
		}
	}

	cout << "The smallest energy is " << energ << " and it was achieved for \
the input vector:\n";

	for (int i = 0; i < n; ++i)
		cout << X[i] << " ";
	cout << "\n";
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
