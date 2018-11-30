#include <iostream>
#include <fstream>
#include <math.h>
#include <random>
#include <vector>
#define mk make_pair
using namespace std;

typedef pair<long long, pair<double, double>> pldd;
typedef pair<vector<double>, vector<pldd>> pvv;

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
	double aux;

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

pvv mcmc(vector<vector<double> > W, vector<double> Y) {
	/* Run the MCMC algorithm to determine the input vector */
	vector<double> X, aux_X, min_X;
	vector<pldd> energies;
	long long total_steps = 0;
	int m = W.size(), n = W[0].size();
	int max_steps = 1000, step = 0;
	double beta = 0.1, beta_step = 0.1, beta_max = 4;
	double energ, aux_energ, min_energ;
	double accept_prob;
	default_random_engine generator;
	uniform_real_distribution<double> unif(0, 1);

	/* Initialize the input vector */
	init(X, n);
	min_X = X;
	min_energ = energ = energy(W, Y, X);
	energies.push_back(mk(0, mk(beta, energ)));

	/* Minimize the energy */
	while (true) {
		++total_steps;
		/* Compute a transition */
		aux_X = transition(X, generator);
		aux_energ = energy(W, Y, aux_X);

		/* Compute the acceptance probabiility */
		accept_prob = min(1.0, exp(-beta * (aux_energ - energ)));

		/* Decide whether to do the transition or not */
		if (unif(generator) <= accept_prob) {
			X = aux_X;
			energ = aux_energ;
			if (energ < min_energ) {
				energies.push_back(mk(total_steps, mk(beta, energ)));
				/* Update minimum values and reset counter */
				min_energ = energ;
				min_X = X;
				step = 0;
			}
			++step;
			/* End the chain if we reached a lower enough energy */
			if (step == max_steps) {
				step = 0;
				X = min_X;
				beta += beta_step;
				if (beta >= beta_max)
					break;
			}
		}
	}

	return mk(min_X, energies);
}

int main(int argc, char **argv) {
	int m, n;
	double x;
	vector<double> Y;
	vector<vector<double> > W;
	ifstream f(argv[1]);
	ofstream g("out.txt");

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

	/* Run the MCMC algorithm */
	auto energies = mcmc(W, Y);

	/* Write the energy evolution and the final prediction */
	for (int i = 0; i < n; ++i)
		g << energies.first[i] << " ";
	g << "\n";
	for (int i = 0; i < energies.second.size(); ++i) {
		g << energies.second[i].first << " " << energies.second[i].second.first;
		g << " " << energies.second[i].second.second << "\n";
	}

	return 0;
}
