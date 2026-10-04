using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IHEPC
{
    internal class Layer
    {
        public double[] Response { get; private set; }
        public double[] AdaptiveMin { get; private set; }
        public double[] AdaptiveMax { get; private set; }
        public double[] DeltaS { get; private set; }

        private readonly List<List<double>> matrix;
        private readonly List<List<double>> derivatives;
        private readonly int[] indexes;
        private readonly double[] offsets;
        private readonly double[] deltaX;

        private readonly int nUrysohns;
        private readonly int nFunctionsInU;
        private readonly int nPoints;

        private double argMin;
        private double argDelta;
        private readonly double targetMin;
        private readonly double targetMax;

        public Layer(int nUrysohns, int nFunctionsInU, int nPoints, double domainMin,
            double domainMax, double fmin, double fmax, Random rng)
        {
            this.nUrysohns = nUrysohns;
            this.nFunctionsInU = nFunctionsInU;
            this.nPoints = nPoints;

            targetMin = fmin;
            targetMax = fmax;

            Response = new double[nUrysohns];
            AdaptiveMin = new double[nUrysohns];
            AdaptiveMax = new double[nUrysohns];
            DeltaS = new double[nUrysohns];

            for (int i = 0; i < nUrysohns; i++)
            {
                AdaptiveMin[i] = fmin;
                AdaptiveMax[i] = fmax;
            }

            matrix = new List<List<double>>(nUrysohns * nFunctionsInU);
            for (int i = 0; i < nUrysohns * nFunctionsInU; i++)
            {
                matrix.Add(new List<double>(new double[nPoints]));
            }

            derivatives = new List<List<double>>(nUrysohns);
            for (int i = 0; i < nUrysohns; i++)
            {
                derivatives.Add(new List<double>(new double[nFunctionsInU]));
            }

            indexes = new int[nUrysohns * nFunctionsInU];
            offsets = new double[nUrysohns * nFunctionsInU];
            deltaX = new double[nFunctionsInU];

            MakeRandomLayer(fmin, fmax, rng);
            AssignMinAndDelta(domainMin, domainMax, nPoints);
        }

        //copy constructor
        public Layer(Layer other)
        {
            nUrysohns = other.nUrysohns;
            nFunctionsInU = other.nFunctionsInU;
            nPoints = other.nPoints;

            argMin = other.argMin;
            argDelta = other.argDelta;
            targetMin = other.targetMin;
            targetMax = other.targetMax;

            Response = (double[])other.Response.Clone();
            AdaptiveMin = (double[])other.AdaptiveMin.Clone();
            AdaptiveMax = (double[])other.AdaptiveMax.Clone();
            DeltaS = (double[])other.DeltaS.Clone();

            indexes = (int[])other.indexes.Clone();
            offsets = (double[])other.offsets.Clone();
            deltaX = (double[])other.deltaX.Clone();

            matrix = new List<List<double>>(other.matrix.Count);

            for (int i = 0; i < other.matrix.Count; ++i)
            {
                matrix.Add(new List<double>(other.matrix[i]));
            }

            derivatives = new List<List<double>>(other.derivatives.Count);

            for (int i = 0; i < other.derivatives.Count; ++i)
            {
                derivatives.Add(new List<double>(other.derivatives[i]));
            }
        }

        public void Forward(double[] input, bool returnAsIs = false)
        {
            for (int k = 0; k < nUrysohns; ++k)
            {
                Response[k] = OneUrysohn(input, k, returnAsIs);
            }
        }

        public void Predict(double[] input, bool returnAsIs = false)
        {
            for (int k = 0; k < nUrysohns; ++k)
            {
                Response[k] = OneUrysohnPredict(input, k, returnAsIs);
            }
        }

        public void MakeDerivativeMatrix(double[] min, double[] max)
        {
            for (int k = 0; k < nFunctionsInU; ++k)
            {
                deltaX[k] = (max[k] - min[k]) / (nPoints - 1);
            }

            for (int k = 0; k < nUrysohns; ++k)
            {
                for (int j = 0; j < nFunctionsInU; ++j)
                {
                    derivatives[k][j] = ComputeDerivative(k * nFunctionsInU + j, deltaX[j]);
                }
            }
        }

        public void BackPropStep(double[] deltasIn, double[] deltasOut)
        {
            for (int j = 0; j < nFunctionsInU; ++j)
            {
                deltasOut[j] = 0.0;
                for (int i = 0; i < nUrysohns; ++i)
                {
                    deltasOut[j] += derivatives[i][j] * deltasIn[i];
                }
                deltasOut[j] /= nUrysohns;
            }
        }

        public void Update()
        {
            for (int k = 0; k < nUrysohns; ++k)
            {
                for (int j = 0; j < nFunctionsInU; ++j)
                {
                    UpdateOne(DeltaS[k], k * nFunctionsInU + j);
                }
            }
        }

        public void NormalizeLayer()
        {
            for (int b = 0; b < nUrysohns; ++b)
            {
                double ymin = AdaptiveMin[b];
                double ymax = AdaptiveMax[b];
                double s = (targetMax - targetMin) / (ymax - ymin);
                double bias = targetMin - s * ymin;

                for (int j = 0; j < nFunctionsInU; ++j)
                {
                    List<double> f = matrix[b * nFunctionsInU + j];

                    for (int k = 0; k < nPoints; ++k)
                    {
                        f[k] = s * f[k] + bias;
                    }
                }

                AdaptiveMin[b] = targetMin;
                AdaptiveMax[b] = targetMax;
            }
        }

        public void Accumulate(Layer layer)
        {
            for (int i = 0; i < matrix.Count; ++i)
            {
                for (int j = 0; j < matrix[i].Count; ++j)
                {
                    matrix[i][j] += layer.matrix[i][j];
                }
            }

            for (int i = 0; i < AdaptiveMin.Length; ++i)
            {
                AdaptiveMin[i] += layer.AdaptiveMin[i];
                AdaptiveMax[i] += layer.AdaptiveMax[i];
            }
        }

        public void Scale(double scale)
        {
            for (int i = 0; i < matrix.Count; ++i)
            {
                for (int j = 0; j < matrix[i].Count; ++j)
                {
                    matrix[i][j] *= scale;
                }
            }

            for (int i = 0; i < AdaptiveMin.Length; ++i)
            {
                AdaptiveMin[i] *= scale;
                AdaptiveMax[i] *= scale;
            }
        }

        public void ShowLayer()
        {
            Console.WriteLine("Urysohn domains");
            for (int i = 0; i < nUrysohns; ++i)
            {
                Console.WriteLine($"{AdaptiveMin[i]}, {AdaptiveMax[i]}");
            }
        }

        private double OneFunction(double x, int id)
        {
            double R = (x - argMin) / argDelta;

            indexes[id] = (int)R;
            offsets[id] = R - indexes[id];

            return matrix[id][indexes[id]] + (matrix[id][indexes[id] + 1] - matrix[id][indexes[id]]) * offsets[id];
        }

        private double OneUrysohn(double[] input, int nWhich, bool returnAsIs = false)
        {
            if (input.Length != nFunctionsInU)
            {
                throw new ArgumentException(
                    String.Format("OneUrysohn: Input size not equal Urysohn size {0} - {1}.", input.Length, nFunctionsInU));
            }

            double r = 0.0;

            for (int j = 0; j < nFunctionsInU; ++j)
            {
                r += OneFunction(
                    input[j],
                    nWhich * nFunctionsInU + j);
            }

            r /= nFunctionsInU;

            if (returnAsIs)
            {
                return r;
            }

            if (r > AdaptiveMax[nWhich])
            {
                AdaptiveMax[nWhich] = r;
            }
            else if (r < AdaptiveMin[nWhich])
            {
                AdaptiveMin[nWhich] = r;
            }

            return (targetMax - targetMin) * (r - AdaptiveMin[nWhich]) / (AdaptiveMax[nWhich] - AdaptiveMin[nWhich])
                + targetMin;
        }

        private double OneUrysohnPredict(double[] input, int nWhich, bool returnAsIs = false)
        {
            if (input.Length != nFunctionsInU)
            {
                throw new ArgumentException(String.Format("Input size not equal Urysohn size {0} - {1}.", input.Length, nFunctionsInU));
            }

            double r = 0.0;
            for (int j = 0; j < nFunctionsInU; ++j)
            {
                r += OneFunction(input[j], nWhich * nFunctionsInU + j);
            }
            r /= nFunctionsInU;

            if (returnAsIs)
            {
                return r;
            }

            r = (targetMax - targetMin) * (r - AdaptiveMin[nWhich]) / (AdaptiveMax[nWhich] - AdaptiveMin[nWhich]) + targetMin;
            if (r < targetMin) return targetMin;
            if (r > targetMax) return targetMax;

            return r;
        }

        private double ComputeDerivative(int id, double delta)
        {
            return (matrix[id][indexes[id] + 1] - matrix[id][indexes[id]]) / delta;
        }

        private void MakeRandomLayer(double fmin, double fmax, Random rnd)
        {
            double mean = (fmax + fmin) / 2.0;
            double range = (fmax - fmin) / 4.0;
            double high = mean + range;
            double low = mean - range;

            for (int i = 0; i < matrix.Count; ++i)
            {
                for (int j = 0; j < matrix[i].Count; ++j)
                {
                    // C++ uniform_real_distribution is [low, high).
                    matrix[i][j] = low + rnd.NextDouble() * (high - low);
                }
            }
        }

        private void AssignMinAndDelta(double domainMin, double domainMax, int nPoints)
        {
            double gap = (domainMax - domainMin) * 0.05;
            argMin = domainMin - gap;
            argDelta = ((domainMax + gap) - argMin) / (nPoints - 1);
        }

        private void UpdateOne(double delta, int id)
        {
            double tmp = delta * offsets[id];
            matrix[id][indexes[id] + 1] += tmp;
            matrix[id][indexes[id]] += delta - tmp;
        }
    }
}