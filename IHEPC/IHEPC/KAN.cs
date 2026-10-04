using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IHEPC
{
    internal class KAN
    {
        private readonly double alpha;
        public readonly List<Layer> layers;

        public KAN(IReadOnlyList<int> Network, IReadOnlyList<int> Grid, double featureMin, double featureMax,
            double intermediateMin, double intermediateMax, double targetMin, double targetMax,
            double learningRate, Random rnd)
        {
            if (Network.Count != Grid.Count + 1)
            {
                throw new ArgumentException(
                    "Network must contain exactly one more " +
                    "element than Grid.");
            }

            int L = Network.Count - 1;
            layers = new List<Layer>(L);

            for (int i = 0; i < L; ++i)
            {
                double argMin = i == 0 ? featureMin : intermediateMin;
                double argMax = i == 0 ? featureMax : intermediateMax;
                double outMin = i == L - 1 ? targetMin : intermediateMin;
                double outMax = i == L - 1 ? targetMax : intermediateMax;

                layers.Add(new Layer(Network[i + 1], Network[i], Grid[i],
                    argMin, argMax, outMin, outMax, rnd));
            }
            alpha = learningRate;
        }

        //copy constructor
        public KAN(KAN other)
        {
            alpha = other.alpha;

            layers = new List<Layer>(other.layers.Count);

            for (int i = 0; i < other.layers.Count; ++i)
            {
                layers.Add(new Layer(other.layers[i]));
            }
        }

        public void Forward(double[] input)
        {
            layers[0].Forward(input);

            for (int i = 1; i < layers.Count; ++i)
            {
                bool lastLayer = layers.Count - 1 == i;

                layers[i].Forward(layers[i - 1].Response, lastLayer);
            }
        }

        public void Predict(double[] input)
        {
            layers[0].Predict(input);

            for (int i = 1; i < layers.Count; ++i)
            {
                bool lastLayer = layers.Count - 1 == i;
                layers[i].Predict(layers[i - 1].Response, lastLayer);
            }
        }

        public double[] GetPrediction()
        {
            int nLayers = layers.Count;
            return layers[nLayers - 1].Response;
        }

        public void Update(double[] targets)
        {
            int nLayers = layers.Count;

            // 1. Making derivatives
            for (int i = nLayers - 1; i > 0; --i)
            {
                layers[i].MakeDerivativeMatrix(layers[i - 1].AdaptiveMin, layers[i - 1].AdaptiveMax);
            }

            // 2. Top deltas
            int nTargets = targets.Length;

            for (int j = 0; j < nTargets; ++j)
            {
                layers[nLayers - 1].DeltaS[j] = (targets[j] - layers[nLayers - 1].Response[j]) * alpha;
            }

            // 3. Other deltas
            for (int i = nLayers - 1; i > 0; --i)
            {
                layers[i].BackPropStep(layers[i].DeltaS, layers[i - 1].DeltaS);
            }

            // 4. Updates
            for (int i = 0; i < nLayers; ++i)
            {
                layers[i].Update();
            }
        }

        public void NormalizeLayers()
        {
            int nLayers = layers.Count;
            for (int i = 0; i < nLayers - 1; ++i)
            {
                layers[i].NormalizeLayer();
            }
        }

        public void Accumulate(KAN kan)
        {
            int nLayers = layers.Count;
            for (int i = 0; i < nLayers; ++i)
            {
                layers[i].Accumulate(kan.layers[i]);
            }
        }

        public void Scale(double scale)
        {
            int nLayers = layers.Count;
            for (int i = 0; i < nLayers; ++i)
            {
                layers[i].Scale(scale);
            }
        }

        public void ShowKAN()
        {
            int nLayers = layers.Count;
            for (int i = 0; i < nLayers; ++i)
            {
                layers[i].ShowLayer();
            }
        }
    }
}

