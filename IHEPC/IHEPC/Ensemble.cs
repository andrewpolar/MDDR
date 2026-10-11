using IHEPC;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IHEPC
{
    class Splitter
    {
        private static readonly Random _random = new Random();
        public IReadOnlyList<double[]> Items { get; }
        public IReadOnlyList<int> Indices { get; }

        public Splitter(IReadOnlyList<double[]> items, IReadOnlyList<int> indices)
        {
            Items = items ?? throw new ArgumentNullException(nameof(items));
            Indices = indices ?? throw new ArgumentNullException(nameof(indices));
        }

        public (Splitter Above, Splitter BelowOrEqual) Split(int mode, int layer)
        {
            if (Indices.Count <= 1)
                throw new InvalidOperationException(
                    "Cannot split a node containing 1 or 0 elements.");

            int vectorSize = Items[Indices[0]].Length;

            double[] unitVector = new double[vectorSize];
            if (0 == mode) unitVector = GenerateNormalUnitVector(vectorSize, layer);
            else if (1 == mode) unitVector = GenerateRandomUnitVector(vectorSize);
            else unitVector = GeneratePrincipalDirectionUnitVector(Items, Indices);

            var innerProducts = new List<(int Index, double Value)>(Indices.Count);
            foreach (int index in Indices)
            {
                double product = ComputeInnerProduct(Items[index], unitVector);
                innerProducts.Add((index, product));
            }

            var sortedProducts = innerProducts.OrderBy(p => p.Value).ToList();
            double median = CalculateMedian(sortedProducts.Select(p => p.Value).ToList());
            var aboveIndices = new List<int>();
            var belowOrEqualIndices = new List<int>();

            foreach (var pair in sortedProducts)
            {
                if (pair.Value > median) aboveIndices.Add(pair.Index);
                else belowOrEqualIndices.Add(pair.Index);
            }

            if (aboveIndices.Count == 0 || belowOrEqualIndices.Count == 0)
            {
                int midPoint = sortedProducts.Count / 2;

                belowOrEqualIndices = sortedProducts
                        .Take(midPoint)
                        .Select(p => p.Index)
                        .ToList();

                aboveIndices = sortedProducts
                        .Skip(midPoint)
                        .Select(p => p.Index)
                        .ToList();
            }
            return (new Splitter(Items, aboveIndices), new Splitter(Items, belowOrEqualIndices));
        }

        private static double[] GenerateRandomUnitVector(int size)
        {
            double[] v = new double[size];
            if (1 == size)
            {
                v[0] = 1.0;
                return v;
            }
            double normSum = 0.0;

            for (int i = 0; i < size; i++)
            {
                double u1 = 1.0 - _random.NextDouble();
                double u2 = 1.0 - _random.NextDouble();

                double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);

                v[i] = randStdNormal;
                normSum += randStdNormal * randStdNormal;
            }

            double magnitude = Math.Sqrt(normSum);

            if (magnitude > 1e-9)
            {
                for (int i = 0; i < size; i++) v[i] /= magnitude;
            }
            return v;
        }

        private double[] GenerateNormalUnitVector(int vectorSize, int layer)
        {
            var vector = new double[vectorSize];
            vector[layer % vectorSize] = 1;
            return vector;
        }

        public static double[] GeneratePrincipalDirectionUnitVector(
            IReadOnlyList<double[]> Items,
            IReadOnlyList<int> Indices)
        {
            if (Items == null || Indices == null)
                throw new ArgumentNullException();

            if (Indices.Count < 2)
                throw new ArgumentException(
                    "At least two vectors are required.");

            int dimension = Items[Indices[0]].Length;

            if (dimension == 0)
                throw new ArgumentException("Vectors cannot be empty.");

            // Calculate the mean vector of the selected observations.
            double[] mean = new double[dimension];

            foreach (int index in Indices)
            {
                double[] x = Items[index];

                if (x.Length != dimension)
                    throw new ArgumentException(
                        "All vectors must have the same dimension.");

                for (int j = 0; j < dimension; j++)
                    mean[j] += x[j];
            }

            for (int j = 0; j < dimension; j++)
                mean[j] /= Indices.Count;

            // Calculate the covariance matrix.
            double[,] cov = new double[dimension, dimension];

            foreach (int index in Indices)
            {
                double[] x = Items[index];

                for (int i = 0; i < dimension; i++)
                {
                    double di = x[i] - mean[i];

                    for (int j = i; j < dimension; j++)
                        cov[i, j] += di * (x[j] - mean[j]);
                }
            }

            double denominator = Indices.Count - 1;

            for (int i = 0; i < dimension; i++)
            {
                for (int j = i; j < dimension; j++)
                {
                    cov[i, j] /= denominator;
                    cov[j, i] = cov[i, j];
                }
            }

            // Find the dominant eigenvector using power iteration.
            // Start with a non-axis-aligned vector.
            double[] v = new double[dimension];

            for (int i = 0; i < dimension; i++)
                v[i] = 1.0 / Math.Sqrt(dimension);

            // A small deterministic perturbation helps avoid an
            // unfortunate initial vector orthogonal to the dominant
            // eigenvector.
            for (int i = 0; i < dimension; i++)
                v[i] += (i + 1.0) * 1e-4;

            Normalize(v);

            double[] next = new double[dimension];

            for (int iteration = 0; iteration < 100; iteration++)
            {
                Array.Clear(next, 0, dimension);

                for (int i = 0; i < dimension; i++)
                    for (int j = 0; j < dimension; j++)
                        next[i] += cov[i, j] * v[j];

                double norm = 0.0;

                for (int i = 0; i < dimension; i++)
                    norm += next[i] * next[i];

                norm = Math.Sqrt(norm);

                // Zero covariance: no preferred direction exists.
                if (norm < 1e-15)
                {
                    for (int i = 0; i < dimension; i++)
                        v[i] = (i == 0) ? 1.0 : 0.0;

                    return v;
                }

                for (int i = 0; i < dimension; i++)
                    next[i] /= norm;

                // Eigenvectors can differ by sign. Compare directions
                // without treating v and -v as different.
                double alignment = 0.0;

                for (int i = 0; i < dimension; i++)
                    alignment += next[i] * v[i];

                if (alignment < 0.0)
                {
                    for (int i = 0; i < dimension; i++)
                        next[i] = -next[i];

                    alignment = -alignment;
                }

                Array.Copy(next, v, dimension);

                if (1.0 - alignment < 1e-12)
                    break;
            }

            return v;
        }

        private static void Normalize(double[] v)
        {
            double norm = 0.0;

            for (int i = 0; i < v.Length; i++)
                norm += v[i] * v[i];

            norm = Math.Sqrt(norm);

            if (norm > 0.0)
                for (int i = 0; i < v.Length; i++)
                    v[i] /= norm;
        }

        private static double ComputeInnerProduct(double[] vecA, double[] vecB)
        {
            double sum = 0.0;
            for (int i = 0; i < vecA.Length; i++) sum += vecA[i] * vecB[i];
            return sum;
        }

        private static double CalculateMedian(List<double> sortedValues)
        {
            int count = sortedValues.Count;
            if (count % 2 == 1) return sortedValues[count / 2];
            return (sortedValues[(count / 2) - 1] + sortedValues[count / 2]) / 2.0;
        }
    }

    class KANLeaf
    {
        public KAN Model { get; }
        public List<int> Indices { get; }

        public KANLeaf(KAN model, IEnumerable<int> indices)
        {
            Model = model ??
                throw new ArgumentNullException(nameof(model));
            Indices = indices.ToList();
        }
    }

    class KANEnsemble
    {
        private readonly List<KANLeaf> _leaves;
        public IReadOnlyList<KANLeaf> Leaves => _leaves;
        public KANEnsemble(List<KANLeaf> leaves)
        {
            _leaves = leaves ??
                throw new ArgumentNullException(nameof(leaves));
        }

        public List<double[]> PredictSample(double[] features)
        {
            if (_leaves.Count == 0)
                throw new InvalidOperationException(
                    "The ensemble contains no models.");

            List<double[]> sample = new List<double[]>();
            for (int leaf = 0; leaf < _leaves.Count; leaf++)
            {
                _leaves[leaf].Model.Predict(features);
                double[] prediction = _leaves[leaf].Model.GetPrediction();
                sample.Add((double[])prediction.Clone());
            }
            return sample;
        }
    }

    class EnsembleBuilder
    {
        private readonly IReadOnlyList<double[]> _features;
        private readonly IReadOnlyList<double[]> _targets;
        private readonly int _epochsPerLayer;

        public EnsembleBuilder(IReadOnlyList<double[]> features, IReadOnlyList<double[]> targets, int epochsPerLayer)
        {
            _features = features ??
                throw new ArgumentNullException(nameof(features));

            _targets = targets ??
                throw new ArgumentNullException(nameof(targets));

            _epochsPerLayer = epochsPerLayer;

            if (_features.Count != _targets.Count)
                throw new ArgumentException(
                    "Features and targets must have the same number of records.");

            if (_epochsPerLayer < 1)
                throw new ArgumentOutOfRangeException(
                    nameof(epochsPerLayer));
        }

        public KANEnsemble Build(KAN rootModel, List<int> rootIndices, int numberOfSplittingLayers, int mode)
        {
            if (rootModel == null)
                throw new ArgumentNullException(nameof(rootModel));

            if (rootIndices == null)
                throw new ArgumentNullException(nameof(rootIndices));

            if (rootIndices.Count == 0)
                throw new ArgumentException(
                    "Root node contains no records.",
                    nameof(rootIndices));

            TrainModel(rootModel, rootIndices, 8); //first model needs longer training
            List<KANLeaf> currentLayer = new List<KANLeaf>
                {
                    new KANLeaf(rootModel, rootIndices)
                };

            for (int layer = 0; layer < numberOfSplittingLayers; ++layer)
            {
                List<KANLeaf> nextLayer = new List<KANLeaf>(currentLayer.Count * 2);
                foreach (KANLeaf parent in currentLayer)
                {
                    List<double[]> residuals = ComputeResiduals(parent.Model, parent.Indices);
                    List<int> localIndices = Enumerable
                            .Range(0, parent.Indices.Count)
                            .ToList();

                    Splitter splitter = new Splitter(residuals, localIndices);
                    (Splitter above, Splitter below) = splitter.Split(mode, layer);

                    List<int> aboveGlobalIndices = above.Indices
                            .Select(local => parent.Indices[local])
                            .ToList();

                    List<int> belowGlobalIndices = below.Indices
                            .Select(local => parent.Indices[local])
                            .ToList();

                    KAN aboveModel = CloneModel(parent.Model);
                    KAN belowModel = CloneModel(parent.Model);

                    TrainModel(aboveModel, aboveGlobalIndices, _epochsPerLayer);
                    TrainModel(belowModel, belowGlobalIndices, _epochsPerLayer);

                    nextLayer.Add(new KANLeaf(aboveModel, aboveGlobalIndices));
                    nextLayer.Add(new KANLeaf(belowModel, belowGlobalIndices));
                }

                currentLayer = nextLayer;
                Console.WriteLine($"Layer {layer + 1}: " + $"{currentLayer.Count} KAN models");
            }

            foreach (KANLeaf leaf in currentLayer)
            {
                TrainModel(leaf.Model, leaf.Indices, _epochsPerLayer);
            }

            Console.WriteLine($"Final training completed for " + $"{currentLayer.Count} KAN models.");
            return new KANEnsemble(currentLayer);
        }

        private void TrainModel(KAN model, IReadOnlyList<int> indices, int epochs)
        {
            for (int epoch = 0; epoch < epochs; ++epoch)
            {
                for (int i = 0; i < indices.Count; ++i)
                {
                    int index = indices[i];
                    model.Forward(_features[index]);
                    model.Update(_targets[index]);
                }
            }
        }

        private List<double[]> ComputeResiduals(KAN model, IReadOnlyList<int> indices)
        {
            List<double[]> residuals = new List<double[]>(indices.Count);

            foreach (int index in indices)
            {
                model.Predict(_features[index]);
                double[] prediction = model.GetPrediction();
                double[] residual = new double[prediction.Length];
                for (int j = 0; j < prediction.Length; ++j)
                {
                    residual[j] = _targets[index][j] - prediction[j];
                }
                residuals.Add(residual);
            }
            return residuals;
        }

        private KAN CloneModel(KAN source)
        {
            return new KAN(source);
        }
    }
}
﻿
