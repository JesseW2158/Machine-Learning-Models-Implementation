#pragma once

#include <cmath>

#include <Eigen/Core>

namespace ActivationFunctions
{
    template <typename InputOutputType = Eigen::VectorXd>
    class BaseIdentity
    {
    public:
        explicit BaseIdentity(int size = 1) {}
        static bool isDerivativeJacobianMatrix() { return false; }
        static std::string getName() { return "Identity"; }
    };

    template <typename InputOutputType = Eigen::VectorXd>
    class IdentityFunction : public BaseIdentity<InputOutputType>
    {
    public:
        const InputOutputType &operator()(const InputOutputType &input) const { return input; }
        double derivative(const InputOutputType &input) const { return InputOutputType::Ones(input.size()); }
    };

    template <>
    class IdentityFunction<double> : public BaseIdentity<double>
    {
    public:
        const double &operator()(const double &input) const { return input; }
        double derivative(const double &input) const { return 1; }
    };

    template <typename InputOutputType = Eigen::VectorXd, typename WeightsType = InputOutputType>
    class BaseSigmoid
    {
    public:
        static bool isDerivativeJacobianMatrix() { return false; }
        static std::string getName() { return "Sigmoid"; }
        InputOutputType getBeta0() { return beta0; }
        InputOutputType getBeta() { return beta; }

    protected:
        InputOutputType beta0;
        InputOutputType beta;
    };

    template <typename InputOutputType = Eigen::VectorXd, typename WeightsType = InputOutputType>
    class SigmoidFunction : public BaseSigmoid<InputOutputType>
    {
    public:
        explicit SigmoidFunction(int size = 1)
        {
            BaseSigmoid<InputOutputType>::beta0 = InputOutputType::Zero(size);
            BaseSigmoid<InputOutputType>::beta = InputOutputType::Ones(size);
        }

        void setParms(const WeightsType &b0, const WeightsType b)
        {
            BaseSigmoid<InputOutputType>::beta0 = b0;
            BaseSigmoid<InputOutputType>::beta = b;
        }

        InputOutputType operator()(const InputOutputType &input)
        {
            if (BaseSigmoid<InputOutputType>::beta0.size() != input.size() || BaseSigmoid<InputOutputType>::beta.size() != input.size())
            {
                BaseSigmoid<InputOutputType>::beta0 = InputOutputType::Zero(input.size());
                BaseSigmoid<InputOutputType>::beta = InputOutputType::Ones(input.size());
            }

            InputOutputType v(input.size());

            for (int i = 0; i < input.size(); ++i)
                v(i) = exp(-(BaseSigmoid<InputOutputType>::beta(i) * input(i) + BaseSigmoid<InputOutputType>::beta0(i)));

            return (InputOutputType::Ones(input.size()) + v).cwiseInverse();
        }
    };
}