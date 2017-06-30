#include <cassert>
#include <iostream>
#include <vector>

#ifndef VECTORHEADERDEF
#define VECTORHEADERDEF

template <class T>
class Vector
{
private:
	std::vector<T> mData; // data stored in vector
	int mSize; // size of vector
public:
	Vector(const Vector<T>& otherVector)
	{
		mSize = otherVector.mSize;
		mData = otherVector.mData;
	}
	Vector(int size)
	{
		assert(size > 0);
		mData.resize(size);
		mSize = size;
	}
	
	int size() const
	{
		return mSize;
	}
	
	T& operator[](int i)// zero-based indexing
	{
		assert(i >= 0);
		assert(i < mSize); //mData.size()
		return mData[i];
	}
	T const& operator[] (int i)const // zero-based indexing
	{
		assert(i >= 0);
		assert(i < mSize);
		return mData[i];
	}
	
	std::vector<T> const& getStorage()
	{
		return mData;
	}
	
	
	Vector<T>& operator=(const Vector<T>& otherVector)
	{
		mData = otherVector.mData;
		return *this;
	}
	
	Vector<T> operator-() const // unary -
	{
		Vector<T> v(mSize);
		for (int i = 0; i<mSize; i++)
		{
			v[i] = -mData[i];
		}
		return v;
	}
	Vector<T> operator+(const Vector<T>& v1) const // binary +
	{
		assert(mSize == v1.mSize);
		Vector<T> v(mSize);
		for (int i = 0; i<mSize; i++)
		{
			v[i] = mData[i] + v1.mData[i];
		}
		return v;
	}
	Vector<T> operator-(const Vector<T>& v1) const // binary -
	{
		assert(mSize == v1.mSize);
		Vector<T> v(mSize);
		for (int i = 0; i<mSize; i++)
		{
			v[i] = mData[i] - v1.mData[i];
		}
		return v;
	}
	
	// scalar multiplication

	Vector<T> operator*(T a) const
	{
		Vector<T> v(mSize);
		for (int i = 0; i<mSize; i++)
		{
			v[i] = a*mData[i];
		}
		return v;
	}
	// p-norm method
	double CalculateNorm(int p = 2) const
	{
		double norm_val, sum = 0.0;
		for (int i = 0; i<mSize; i++)
		{
			sum += pow(fabs(mData[i]), p);
		}
		norm_val = pow(sum, 1.0 / ((double)(p)));
		return norm_val;
	}
};


#endif