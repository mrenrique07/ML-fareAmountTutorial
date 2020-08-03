using System;
using System.IO;
using Microsoft.ML;

namespace TaxiFarePrediction
{
    class Program
    {
        //ruta de los modelos y _textloader
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 0);
            var model = Train(mlContext, _trainDataPath);
            Evaluate(mlContext, model);
            TestSinglePrediction(mlContext, model);

        }
        public static ITransformer Train(MLContext mlContext, string dataPath)
        {
            //Load and transform data
            IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(dataPath, hasHeader: true, separatorChar: ',');
            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
                .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripDistance", "PaymentTypeEncoded"))
                //Choose a learning algorithm
                .Append(mlContext.Regression.Trainers.FastTree());

            //Train the model
            var model = pipeline.Fit(dataView);

            return model;

        }

        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(_testDataPath, hasHeader: true, separatorChar: ',');
            var predictions = model.Transform(dataView);
            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");
            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");
        }

        private static void TestSinglePrediction(MLContext mlContext, ITransformer model)
        {
            var predictionFunction = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(model);
            
            //var taxiTripSample = new TaxiTrip();

            //Console.WriteLine( "Insert ID: ");
            //taxiTripSample.VendorId = Console.ReadLine();

            //Console.WriteLine("insert rate code:");
            //taxiTripSample.RateCode = Console.ReadLine();

            //Console.WriteLine("Insert Passeger Count: ");
            //taxiTripSample.PassengerCount = float.Parse(Console.ReadLine());

            //Console.WriteLine("Insert Trip Time: ");
            //taxiTripSample.TripTime = float.Parse(Console.ReadLine());

            //Console.WriteLine("Insert Trip Distance: ");
            //taxiTripSample.TripDistance = float.Parse(Console.ReadLine());

            //Console.WriteLine("Insert Payment type, 1 cash, 2 card: ");
            //int Evaluar = Convert.ToInt32 (Console.ReadLine());
            //#region Evalua e inserta
            //int i = 1;

            //do{
            //    if (Evaluar > 0 && Evaluar < 2)
            //    {
            //        taxiTripSample.PaymentType = "CSH";

            //    }
            //    else if (Evaluar > 1 && Evaluar < 3)
            //    {
            //        taxiTripSample.PaymentType = "CRD";

            //    }
            //    else
            //    {
            //        Console.WriteLine("Please input a valid option");
            //        i = 0;
            //    }
            //} while (i == 0);
            //#endregion

            //Console.WriteLine("Enter fare amount: ");
            //taxiTripSample.FareAmount = float.Parse( Console.ReadLine());

            var taxiTripSample = new TaxiTrip()
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1140,
                TripDistance = 3.75f,
                PaymentType = "CRD",
                FareAmount = 0 // To predict. Actual/Observed = 15.5
            };

            var prediction = predictionFunction.Predict(taxiTripSample);

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
            Console.WriteLine($"**********************************************************************");
        }
    }
}
