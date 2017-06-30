#include <iostream>
#include <cmath>
#include "CalculateExponential.hpp"

void CalculateExponential(ComplexNumber **A, int nMax, ComplexNumber **res)
{
	ComplexNumber **temp1 = new ComplexNumber *[3];
	ComplexNumber **temp2 = new ComplexNumber *[3];
	ComplexNumber **temp3 = new ComplexNumber *[3];
	ComplexNumber **I = new ComplexNumber *[3];
	for (int i = 0; i < 3; i++)
	{
		temp1[i] = new ComplexNumber[3];
		temp2[i] = new ComplexNumber[3];
		temp3[i] = new ComplexNumber[3];
		I[i] = new ComplexNumber[3];
	}
	// creating identity matrix
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			if (i == j)
			{
				I[i][j] = 1.0;
			}
			else
			{
				I[i][j] = 0.0;
			}
		}
	}

	if (nMax > 1)
	{
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				temp1[i][j] = A[i][j];
			}
		}
		for (int a = 2; a <= nMax; a++)
		{
			double m = (double)(a);
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					temp2[i][j] = 0;
					for (int k = 0; k < 3; k++)
					{
						temp2[i][j] = temp2[i][j] + (temp1[i][k] * A[k][j]);
					}
				}
			}
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					temp1[i][j] = temp2[i][j]*(1/m);
					temp3[i][j] = temp3[i][j] + temp1[i][j];

				}
			}
		}
	}

	// result = I+A
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			if (nMax > 1)
			{
				res[i][j] = I[i][j] + A[i][j] + temp3[i][j];
			}
			else if (nMax==0)
			{
				res[i][j] = I[i][j] + A[i][j];
			}
			else
			{
				res[i][j] = I[i][j] + A[i][j];
			}
		}
	}
	for (int i = 0; i < 3; i++)
	{
		delete[] temp1[i];
		delete[] temp2[i];
		delete[] temp3[i];
		delete[] I[i];
	}
	delete[] temp1;
	delete[] temp2;
	delete[] temp3;
	delete[] I;
}


//non mandatory
void printMatrix(ComplexNumber **A, int rows, int cols)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			std::cout << A[i][j] << " ";
			if (j == cols - 1)
			{
				std::cout << "\n";
			}
		}
	}
}



