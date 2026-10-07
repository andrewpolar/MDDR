// ============================================================================
// KANKAN (Kolmogorov-Arnold Networks KAczmarz-Newton)
// MDDR (Multivariate Divisive Data Resorting)
//
// Concept: Andrew Polar, Mike Poluektov
// Theory and algorithms are described in peer-reviewed publications.
// https://www.sciencedirect.com/science/article/abs/pii/S0016003220301149
// https://www.sciencedirect.com/science/article/abs/pii/S0952197620303742
// https://link.springer.com/article/10.1007/s10994-025-06800-6
// https://www.mdpi.com/2673-3951/6/3/88
// https://www.sciencedirect.com/science/article/pii/S0925231226021703
// https://arxiv.org/abs/2609.23906
// Additional information, examples and documentation are available at
// OpenKAN.org.
//
// ---------------------------------------------------------------------------
// License
//
// If an end user somehow earns billions of US dollars using this software
// and later encounters the developer asking for spare change outside a
// McDonald's, the end user is under no obligation to buy the developer
// a sandwich.
//
// Symmetry Clause
//
// Likewise, if the developer becomes rich and famous by publishing this
// software and later meets an unfortunate end user who went bankrupt while
// using it, the developer is also under no obligation to buy the end user
// a sandwich.
//----------------------------------------------------------------------------

// This is test of concept from https://arxiv.org/abs/2609.23906
// on the observation of social system. 
// Dataset https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

namespace IHEPC
{
    class Program
    {
        static (double[] CentralPoint, double Radius) CalculateCalibration(
            IReadOnlyList<double[]> sample, double probability)
        {
            if (sample == null || sample.Count == 0)
                throw new ArgumentException("Sample cannot be empty.");

            if (probability < 0.0 || probability > 1.0)
                throw new ArgumentOutOfRangeException(nameof(probability));

            int dimensions = sample[0].Length;

            // Component-wise median
            double[] centralPoint = new double[dimensions];

            for (int d = 0; d < dimensions; d++)
            {
                List<double> values = sample.Select(p => p[d]).ToList();
                values.Sort();
                int middle = values.Count / 2;

                centralPoint[d] = values.Count % 2 == 1
                        ? values[middle]
                        : (values[middle - 1] + values[middle]) / 2.0;
            }

            // Distances from the central point
            List<double> distances = new List<double>(sample.Count);
            foreach (double[] point in sample)
            {
                double sum = 0.0;
                for (int d = 0; d < dimensions; d++)
                {
                    double diff = point[d] - centralPoint[d];
                    sum += diff * diff;
                }
                distances.Add(Math.Sqrt(sum));
            }
            distances.Sort();

            int count = (int)Math.Ceiling(probability * sample.Count);

            double radius = count <= 0 ? 0.0
                : distances[Math.Min(count - 1, distances.Count - 1)];

            return (centralPoint, radius);
        }

        static void Main(string[] args)
        {
            //dataset choices
            const int nPrevRecords = 15;
            const int nAggregation = 15;

            //first deterministic model for features reduction
            const int nHidden = 8;
            const int nPoints0 = 20;
            const int nPoints1 = 10;

            //training parameters
            const double alpha = 0.1;
            const int nEpochs = 16;

            //ensemble settings 
            const int nIntermediate = 8;
            const int nPoints0Ensemble = 24;
            const int nPoints1Ensemble = 16;
            const int nEpochsInSpit = 3;
            const int nTreeDepth = 8;

            //data preparation
            DataHolder dh = new DataHolder();
            dh.ReadData();
            dh.MakeDatasets(nAggregation);
            int nDataSize = dh.GetDataSetSize();
            int nFeatures = dh.GetNumberOfFeatures();
            int nTargets = dh.GetNumberOfTargets();
            double Min = dh.GetMin();
            double Max = dh.GetMax();

            //deterministic model
            IReadOnlyList<int> network = new List<int> { nPrevRecords * nFeatures, nHidden, nTargets };
            IReadOnlyList<int> points = new List<int> { nPoints0, nPoints1 };
            KAN kan = new KAN(network, points, Min, Max, Min, Max, Min, Max, alpha, new Random());
            Console.WriteLine("Training {0} epochs...", nEpochs);
            for (int epoch = 0; epoch < nEpochs; ++epoch)
            {
                for (int targetIndex = nPrevRecords * nFeatures; targetIndex < nDataSize / 2; ++targetIndex)
                {
                    (double[] features, double[] targets) = dh.GetTrainingData(targetIndex, nPrevRecords);
                    kan.Forward(features);
                    kan.Update(targets);
                }
                Console.Write("{0} ", epoch + 1);
            }
            Console.WriteLine();

            //feature reduction
            List<double[]> hiddenList = new List<double[]>();
            List<double[]> targetList = new List<double[]>();
            for (int targetIndex = nPrevRecords * nFeatures; targetIndex < nDataSize / 2; ++targetIndex)
            {
                (double[] features, double[] targets) = dh.GetTrainingData(targetIndex, nPrevRecords);
                kan.layers[0].Predict(features);
                double[] hidden = (double[])kan.layers[0].Response.Clone();
                hiddenList.Add(hidden);
                targetList.Add(targets);
            }
            Console.WriteLine("Data for MDDR model is ready");

            //MDDR model
            int N = hiddenList.Count;
            IReadOnlyList<int> MDDRnetwork = new List<int> { nHidden, nIntermediate, nTargets };
            IReadOnlyList<int> MDDRpoints = new List<int> { nPoints0Ensemble, nPoints1Ensemble };
            KAN rootKan = new KAN(MDDRnetwork, MDDRpoints, Min, Max, Min, Max, Min, Max, alpha, new Random());
            List<int> indices = Enumerable.Range(0, N).ToList();
            EnsembleBuilder builder = new EnsembleBuilder(hiddenList, targetList, nEpochsInSpit);
            KANEnsemble ensemble = builder.Build(rootKan, indices, nTreeDepth);

            //calibration
            Console.WriteLine("Calibration...");
            for (int step = 1; step <= 9; ++step)
            {
                double P = step * 0.10;
                int within = 0;
                int total = 0;
                for (int targetIndex = nDataSize / 2; targetIndex < nDataSize / 2 + nDataSize / 4; ++targetIndex)
                {
                    (double[] features, double[] targets) = dh.GetTrainingData(targetIndex, nPrevRecords);
                    kan.layers[0].Predict(features);
                    double[] hidden = (double[])kan.layers[0].Response.Clone();
                    List<double[]> sample = ensemble.PredictSample(hidden);
                    (double[] point, double radius) = CalculateCalibration(sample, P);
                    double distance = 0.0;
                    for (int j = 0; j < point.Length; ++j)
                    {
                        distance += (point[j] - targets[j]) * (point[j] - targets[j]);
                    }
                    distance = Math.Sqrt(distance);
                    if (distance <= radius)
                    {
                        ++within;
                    }
                    ++total;
                }
                Console.WriteLine("Target {0:0.00}, actual {1:0.00}", P, (double)within / (double)total);
            }

            //calibration
            Console.WriteLine("Testing...");
            for (int step = 1; step <= 9; ++step)
            {
                double P = step * 0.10;
                int within = 0;
                int total = 0;
                for (int targetIndex = nDataSize / 2 + nDataSize / 4; targetIndex < nDataSize; ++targetIndex)
                {
                    (double[] features, double[] targets) = dh.GetTrainingData(targetIndex, nPrevRecords);
                    kan.layers[0].Predict(features);
                    double[] hidden = (double[])kan.layers[0].Response.Clone();
                    List<double[]> sample = ensemble.PredictSample(hidden);
                    (double[] point, double radius) = CalculateCalibration(sample, P);
                    double distance = 0.0;
                    for (int j = 0; j < point.Length; ++j)
                    {
                        distance += (point[j] - targets[j]) * (point[j] - targets[j]);
                    }
                    distance = Math.Sqrt(distance);
                    if (distance <= radius)
                    {
                        ++within;
                    }
                    ++total;
                }
                Console.WriteLine("Target {0:0.00}, actual {1:0.00}", P, (double)within / (double)total);
            }
        }
    }
}
