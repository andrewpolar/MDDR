using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace IHEPC
{
    class Record
    {
        public DateTime date { get; set; }
        public double active { get; set; }
        public double reactive { get; set; }
        public double voltage { get; set; }
        public double intensity { get; set; }
        public double s1 { get; set; }
        public double s2 { get; set; }
        public double s3 { get; set; }
    }

    internal class DataHolder
    {
        private List<Record> records = new List<Record>();
        public List<double[]> dataset = new List<double[]>();
        private const string fileName = "../../../../DATA/household_power_consumption.txt";
        public const int nFeatures = 10;
        public const int nTargets = 7;
        public const double Min = 0.0;
        public const double Max = 100.0;

        public void ReadData()
        {
            Console.WriteLine("Reading data...");
            using (StreamReader reader = new StreamReader(fileName))
            {
                // Skip header
                reader.ReadLine();
                while (true)
                {
                    string line = reader.ReadLine();
                    if (line == null) break;
                    string[] parts = line.Split(';');

                    // Skip invalid rows
                    if (parts.Length != 9) continue;

                    // Skip rows with missing values
                    if (Array.Exists(parts, p => p == "?")) continue;

                    Record record = new Record
                    {
                        date = DateTime.ParseExact(
                            Validator(parts[0]) + " " + parts[1],
                            "dd/MM/yyyy HH:mm:ss",
                            CultureInfo.InvariantCulture
                        ),
                        active = double.Parse(parts[2], CultureInfo.InvariantCulture),
                        reactive = double.Parse(parts[3], CultureInfo.InvariantCulture),
                        voltage = double.Parse(parts[4], CultureInfo.InvariantCulture),
                        intensity = double.Parse(parts[5], CultureInfo.InvariantCulture),
                        s1 = double.Parse(parts[6], CultureInfo.InvariantCulture),
                        s2 = double.Parse(parts[7], CultureInfo.InvariantCulture),
                        s3 = double.Parse(parts[8], CultureInfo.InvariantCulture)
                    };
                    records.Add(record);
                }
            }

            Console.WriteLine($"Records loaded: {records.Count}");
        }

        private string Validator(string input)
        {
            string[] s = input.Split('/');
            if (1 == s[0].Length)
            {
                s[0] = "0" + s[0];
            }
            if (1 == s[1].Length)
            {
                s[1] = "0" + s[1];
            }
            return $"{s[0]}/{s[1]}/{s[2]}";
        }

        public void MakeDatasets(int aggregation)
        {
            Console.WriteLine("Making dataset...");
            double minActive = records[0].active;
            double minReactive = records[0].reactive;
            double minVoltage = records[0].voltage;
            double minIntensity = records[0].intensity;
            double minS1 = records[0].s1;
            double minS2 = records[0].s2;
            double minS3 = records[0].s3;

            double maxActive = minActive;
            double maxReactive = minReactive;
            double maxVoltage = minVoltage;
            double maxIntensity = minIntensity;
            double maxS1 = minS1;
            double maxS2 = minS2;
            double maxS3 = minS3;

            double[] monthLookup = new double[]
            {
                10.0, 10.0, 20.0, 30.0, 40.0, 50.0, 50.0, 60.0, 70.0, 80.0, 90.0, 99.0
            };
            double[] dayLookup = new double[]
            {
                20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0
            };

            foreach (var record in records)
            {
                if (record.active < minActive) { minActive = record.active; }
                if (record.active > maxActive) { maxActive = record.active; }

                if (record.reactive < minReactive) { minReactive = record.reactive; }
                if (record.reactive > maxReactive) { maxReactive = record.reactive; }

                if (record.voltage < minVoltage) { minVoltage = record.voltage; }
                if (record.voltage > maxVoltage) { maxVoltage = record.voltage; }

                if (record.intensity < minIntensity) { minIntensity = record.intensity; }
                if (record.intensity > maxIntensity) { maxIntensity = record.intensity; }

                if (record.s1 < minS1) { minS1 = record.s1; }
                if (record.s1 > maxS1) { maxS1 = record.s1; }

                if (record.s2 < minS2) { minS2 = record.s2; }
                if (record.s2 > maxS2) { maxS2 = record.s2; }

                if (record.s3 < minS3) { minS3 = record.s3; }
                if (record.s3 > maxS3) { maxS3 = record.s3; }
            }

            double[] x = new double[nFeatures];
            Array.Clear(x);
            int cnt = 0;
            for (int i = 0; i < records.Count; ++i)
            {               
                x[0] += (records[i].active - minActive) / (maxActive - minActive) * (Max - Min) + Min;
                x[1] += (records[i].reactive - minReactive) / (maxReactive - minReactive) * (Max - Min) + Min;
                x[2] += (records[i].voltage - minVoltage) / (maxVoltage - minVoltage) * (Max - Min) + Min;
                x[3] += (records[i].intensity - minIntensity) / (maxIntensity - minIntensity) * (Max - Min) + Min;
                x[4] += (records[i].s1 - minS1) / (maxS1 - minS1) * (Max - Min) + Min;
                x[5] += (records[i].s2 - minS2) / (maxS2 - minS2) * (Max - Min) + Min;
                x[6] += (records[i].s3 - minS3) / (maxS3 - minS3) * (Max - Min) + Min;
                x[7] += monthLookup[records[i].date.Month - 1]; 
                x[8] += dayLookup[(int)records[i].date.DayOfWeek];
                x[9] += Math.Round((records[i].date.Hour * 60 + records[i].date.Minute) / 15.0);

                if (++cnt >= aggregation)
                {
                    for (int j = 0; j < x.Length; ++j)
                    {
                        x[j] /= aggregation;
                    }
                    dataset.Add(x);
                    x = new double[nFeatures];
                    Array.Clear(x);
                    cnt = 0;
                }  
            }

            records.Clear();
            Console.WriteLine($"Dataset:        {dataset.Count}");
        }

        public (double[] features, double[] target) GetTrainingData(int index, int previousRecords)
        {
            double[] features = new double[previousRecords * nFeatures];
            double[] target = new double[nTargets];

            int k = 0;
            for (int i = index - previousRecords; i < index; ++i)
            {
                for (int j = 0; j < nFeatures; ++j)
                {
                    features[k++] = dataset[i][j];
                }
            }
            for (int j = 0; j < nTargets; ++j)
            {
                target[j] = dataset[index][j];
            }
            return (features, target);
        }

        public int GetDataSetSize()
        {
            return dataset.Count;
        }

        public int GetNumberOfFeatures()
        {
            return nFeatures;
        }

        public int GetNumberOfTargets()
        {
            return nTargets;
        }

        public double GetMin()
        {
            return Min;
        }

        public double GetMax()
        {
            return Max;
        }
    }
}
