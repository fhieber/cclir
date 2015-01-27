/*
 * lossfunction-test.cpp
 *
 *  Created on: Apr 23, 2013
 */

#include "losses.h"



int main(int argc, char** argv) {
	NDCG n(false);
	HammingDistance hd(false);
	MAP map(false);
	PlackettLuce pl(false);
	Permutation y(10);
	y[0]=4; y[1]=6; y[2]=2; y[3]=7; y[4]=9; y[5]=8; y[6]=0; y[7]=1; y[8]=3; y[9]=5;
	cout << "NDCG\t" << n(y,y) << endl;
	cout << "MAP\t" << map(y,y) << endl;
	cout << "Hamming\t" << hd(y,y) << endl;
	cout << "PlackettLuce\t" << pl(y,y) << endl;
	Permutation x(10);
	x[0]=4; x[1]=2; x[2]=5; x[3]=3; x[4]=0; x[5]=1; x[6]=6; x[7]=7; x[8]=8; x[9]=9;
	cout << "NDCG\t" << n(x,y) << endl;
	cout << "MAP\t" << map(x,y) << endl;
	cout << "Hamming\t" << hd(x,y) << endl;
	cout << "PlackettLuce\t" << pl(x,y) << endl;
	x[0]=0; x[1]=1; x[2]=2; x[3]=3; x[4]=4; x[5]=5; x[6]=6; x[7]=7; x[8]=8; x[9]=9;
	cout << "NDCG\t" << n(x,y) << endl;
	cout << "MAP\t" << map(x,y) << endl;
	cout << "Hamming\t" << hd(x,y) << endl;
	cout << "PlackettLuce\t" << pl(x,y) << endl;
	x[0]=5; x[1]=4; x[2]=2; x[3]=7; x[4]=6; x[5]=8; x[6]=0; x[7]=1; x[8]=3; x[9]=9;
	cout << "NDCG\t" << n(x,y) << endl;
	cout << "MAP\t" << map(x,y) << endl;
	cout << "Hamming\t" << hd(x,y) << endl;
	cout << "PlackettLuce\t" << pl(x,y) << endl;
	x[0]=4; x[1]=6; x[2]=7; x[3]=2; x[4]=9; x[5]=0; x[6]=8; x[7]=1; x[8]=5; x[9]=3;
	cout << "NDCG\t" << n(x,y) << endl;
	cout << "MAP\t" << map(x,y) << endl;
	cout << "Hamming\t" << hd(x,y) << endl;
	cout << "PlackettLuce\t" << pl(x,y) << endl;
}

