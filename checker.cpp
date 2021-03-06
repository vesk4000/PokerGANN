#include <iostream>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>

const int MAX_N = 50;

int n;
float ante;
float betSize;

float openBetProbs[MAX_N];
float checkBetProbs[MAX_N];
float betCallProbs[MAX_N];
float checkBetCallProbs[MAX_N];

bool oppOpenBet[MAX_N];
bool oppCheckBet[MAX_N];
bool oppBetCall[MAX_N];
bool oppCheckBetCall[MAX_N];

float firstExpProfits[MAX_N];
float secondExpProfits[MAX_N];
float expProfit;

void error(const std::string& message)
{
	std::cerr << message << std::endl;
	std::cout << 0 << std::endl;
	exit(0);
}

float checkCheck(int self, int opponent)
{
	if (self < opponent) return 0;
	else return ante; 
}

float betCall(int self, int opponent)
{
	if (self < opponent) return -betSize;
	else return (ante + betSize); 
}


void findExpProfit()
{
	expProfit = 0;
	for (int i = 0; i < n; ++i)
	{
		float oppCheckBetCallExp = 0;
		float oppCheckBetFoldExp = 0;

		for (int j = 0; j < n; ++j)
		{
			if (i == j) continue;

		   oppCheckBetCallExp += checkBetProbs[j] * betCall(i, j);
		}

		oppCheckBetCall[i] = oppCheckBetCallExp > oppCheckBetFoldExp;

		float oppOpenBetExp = 0;
		float oppOpenCheckExp = 0;

		for (int j = 0; j < n; ++j)
		{
			if (i == j) continue;

			oppOpenBetExp += (1 - betCallProbs[j]) * ante;
			oppOpenBetExp += betCallProbs[j] * betCall(i, j);

			oppOpenCheckExp += (1 - checkBetProbs[j]) * checkCheck(i, j);
			if(oppCheckBetCall[i])
				oppOpenCheckExp += checkBetProbs[j] * betCall(i, j);
			else
				oppOpenCheckExp += checkBetProbs[j] * 0;
		}

		oppOpenBet[i] = oppOpenBetExp > oppOpenCheckExp;

		float oppCheckBetExp = 0;
		float oppCheckCheckExp = 0;

		for (int j = 0; j < n; ++j)
		{
			if (i == j) continue;

			oppCheckBetExp += (1 - openBetProbs[j]) * (1 - checkBetCallProbs[j]) * ante;
			oppCheckBetExp += (1 - openBetProbs[j]) * checkBetCallProbs[j] * betCall(i, j);

			oppCheckCheckExp += (1 - openBetProbs[j]) * checkCheck(i, j);
		}

		oppCheckBet[i] = oppCheckBetExp > oppCheckCheckExp;

		float oppBetCallExp = 0;
		float oppBetFoldExp = 0;

		for (int j = 0; j < n; ++j)
		{
			if (i == j) continue;

			oppBetCallExp += openBetProbs[j] * betCall(i, j);
		}

		oppBetCall[i] = oppBetCallExp > oppBetFoldExp;
		if(oppBetCall[i])
			expProfit++;
	}

	//expProfit = 0;

	for (int i = 0; i < n; ++i)
	{
		firstExpProfits[i] = 0;
		secondExpProfits[i] = 0;

		for (int j = 0; j < n; ++j)
		{
			if (i == j) continue;

			if(oppBetCall[j])
				firstExpProfits[i] += 1;
			// 	firstExpProfits[i] += openBetProbs[i] * betCall(i, j);
			// else
			// 	firstExpProfits[i] += openBetProbs[i] * ante;
			/*if(oppCheckBet[j]) {
				firstExpProfits[i] += (1 - openBetProbs[i]) * (1 - checkBetCallProbs[i]) * 0;
				firstExpProfits[i] += (1 - openBetProbs[i]) * checkBetCallProbs[i] * betCall(i, j);
			}
			else
				firstExpProfits[i] += (1 - openBetProbs[i]) * checkCheck(i, j);*/

			if(oppOpenBet[j]) {
				secondExpProfits[i] += oppOpenBet[j] * (1 - betCallProbs[i]) * 0;
				secondExpProfits[i] += betCallProbs[i] * betCall(i, j);
			}
			else {
				secondExpProfits[i] += (1 - checkBetProbs[i]) * checkCheck(i, j);
				if(oppCheckBetCall[j])
					secondExpProfits[i] += checkBetProbs[i] * betCall(i, j);
				else
					secondExpProfits[i] += checkBetProbs[i] * ante;
				
			}

		}

		//firstExpProfits[i] /= (n - 1);
		secondExpProfits[i] /= (n - 1);

		//expProfit += firstExpProfits[i];
		//expProfit += secondExpProfits[i];
	}

	//expProfit ///= /*2 **/ n;
}

void info()
{
	float avgOpenBetProb = 0;
	float avgBetCallProb = 0;
	float avgCheckBetProb = 0;
	float avgCheckBetCallProb = 0;

	for (int i = 0; i < n; ++i)
	{
		avgOpenBetProb += openBetProbs[i] / n;
		avgBetCallProb += betCallProbs[i] / n;
		avgCheckBetProb += checkBetProbs[i] / n;
		avgCheckBetCallProb += checkBetCallProbs[i] / n;
	}

	std::cerr << std::endl;
	std::cerr << "Average bet/call probabilities breakdown:" << std::endl;
	std::cerr << "open: " << std::fixed << std::setprecision(4) << avgOpenBetProb << std::endl;
	std::cerr << "check: " << std::fixed << std::setprecision(4) << avgCheckBetProb << std::endl;
	std::cerr << "bet: " << std::fixed << std::setprecision(4) << avgBetCallProb << std::endl;
	std::cerr << "check-bet: " << std::fixed << std::setprecision(4) << avgCheckBetCallProb << std::endl;

	std::cerr << std::endl;
	std::cerr << "Profit breakdown:" << std::endl;

	for (int i = 0; i < n; ++i)
	{
		std::cerr << i + 1;
		if (i + 1 < 10) std::cerr << " ";
		std::cerr << "  " << std::fixed << std::setprecision(4) << firstExpProfits[i];
		std::cerr << "  " << std::fixed << std::setprecision(4) << secondExpProfits[i];
		std::cerr << std::endl;
	}

	std::cerr << std::endl;
	std::cerr << "Opponent strategy:" << std::endl;
	std::cerr << "card  open   check  bet    check-bet" << std::endl;

	for (int i = 0; i < n; ++i)
	{
		std::cerr << i + 1 << "  ";
		if (i + 1 < 10) std::cerr << " ";
		std::cerr << "  " << (oppOpenBet[i] ? "bet  " : "check");
		std::cerr << "  " << (oppCheckBet[i] ? "bet  " : "check");
		std::cerr << "  " << (oppBetCall[i] ? "call " : "fold ");
		std::cerr << "  " << (oppCheckBetCall[i] ? "call " : "fold ");
		std::cerr << std::endl;
	}
}

void parseStrategy(std::ifstream& stratFile)
{
	std::string line;
	std::string word;

	int card;
	std::string situation;
	std::string action;
	float prob;

	std::fill(openBetProbs, openBetProbs + n, -1);
	std::fill(checkBetProbs, checkBetProbs + n, -1);
	std::fill(betCallProbs, betCallProbs + n, -1);
	std::fill(checkBetCallProbs, checkBetCallProbs + n, -1);

	while (!stratFile.eof())
	{
		std::getline(stratFile, line);
		std::stringstream lineStream(line);

		word = "";
		lineStream >> word;
		if (word == "") continue;

		card = std::stoi(word) - 1;
		
		word = "";
		lineStream >> word;
		if (word == "") error("Unexpected line end in.");
	
		situation = word;
		if (situation.back() != ':') error("Expected ':' after situation.");
		situation = situation.substr(0, situation.size() - 1);
		
		word = "";
		lineStream >> word;
		if (word == "") error("Unexpected line end.");
	
		action = word;

		word = "";
		lineStream >> word;

		if (word == "") prob = 1;
		else prob = std::stod(word);

		word = "";
		lineStream >> word;
		if (word != "") error("Expected line end.");

		if (card < 0 || card > n) error("Invalid card number.");

		if (situation != "open" && situation != "check" && situation != "bet" && situation != "check-bet") error("Invalid situation.");

		float* toSet = nullptr;

		if (situation == "open" && (action == "check" || action == "bet")) toSet = &openBetProbs[card];
		else if (situation == "check" && (action == "check" || action == "bet")) toSet = &checkBetProbs[card];
		else if (situation == "bet" && (action == "fold" || action == "call")) toSet = &betCallProbs[card];
		else if (situation == "check-bet" && (action == "fold" || action == "call")) toSet = &checkBetCallProbs[card];
		else error("Invalid action.");

		if (prob < 0 || prob > 1) error("Invalid probability.");

		if (*toSet >= 0) error("Overspecified card-situation pair.");

		if (action == "check" || action == "fold") *toSet = 1 - prob;
		else if (action == "bet" || action == "call") *toSet = prob;
		else error("Internal error.");
	}

	for (int i = 0; i < n; ++i)
	{
		if (openBetProbs[i] < 0 || checkBetProbs[i] < 0 || betCallProbs[i] < 0 || checkBetCallProbs[i] < 0) error("Unspecified card-situation pair.");
	}
}

int main(int argc, char *argv[])
{
	std::ifstream in(argv[1]);
	std::ifstream out(argv[3]);

	if (!in.is_open())
	{
		std::cerr << "In-file " << argv[2] << " not found." << std::endl;
		std::cout << 0 << std::endl;
		return 0;
	}

	if (!out.is_open())
	{
		std::cerr << "Out-file " << argv[3] << " not found." << std::endl;
		std::cout << 0 << std::endl;
		return 0;
	}

	in >> n >> ante >> betSize;

	parseStrategy(out);

	findExpProfit();

	float score = std::min(std::max(2 * expProfit / (0.99 * ante), 0.0), 1.0);
	score = score * score * score * score * score;

	//std::cout << std::fixed << std::setprecision(4) << score << std::endl;
	std::cout/* << "Average profit: "*/ << std::fixed << std::setprecision(20) << expProfit << std::endl;

	//info();

	return 0;
}
