// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

using System;
using Microsoft.Research.SEAL;
using System.Collections.Generic;

namespace SEALNetExamples
{
    partial class Examples
    {
        /*
        Both the BFV scheme (with BatchEncoder) as well as the CKKS scheme support native
        vectorized computations on encrypted numbers. In addition to computing slot-wise,
        it is possible to rotate the encrypted vectors cyclically.
        BFV方案（带有Batch编码器）以及CKS方案支持对加密数字的本机矢量化计算。
		除了计算时隙，还可以循环地旋转加密的矢量。
        */
        private static void ExampleRotationBFV()
        {
            Utilities.PrintExampleBanner("Example: Rotation / Rotation in BFV");

            EncryptionParameters parms = new EncryptionParameters(SchemeType.BFV);

            ulong polyModulusDegree = 8192;
            parms.PolyModulusDegree = polyModulusDegree;
            parms.CoeffModulus = CoeffModulus.BFVDefault(polyModulusDegree);
            parms.PlainModulus = PlainModulus.Batching(polyModulusDegree, 20);

            SEALContext context = new SEALContext(parms);
            Utilities.PrintParameters(context);
            Console.WriteLine();

            KeyGenerator keygen = new KeyGenerator(context);
            PublicKey publicKey = keygen.PublicKey;
            SecretKey secretKey = keygen.SecretKey;
            RelinKeys relinKeys = keygen.RelinKeys();
            Encryptor encryptor = new Encryptor(context, publicKey);
            Evaluator evaluator = new Evaluator(context);
            Decryptor decryptor = new Decryptor(context, secretKey);

            BatchEncoder batchEncoder = new BatchEncoder(context);
            ulong slotCount = batchEncoder.SlotCount;
            ulong rowSize = slotCount / 2;
            Console.WriteLine($"Plaintext matrix row size: {rowSize}");

            ulong[] podMatrix = new ulong[slotCount];
            podMatrix[0] = 0;
            podMatrix[1] = 1;
            podMatrix[2] = 2;
            podMatrix[3] = 3;
            podMatrix[rowSize] = 4;
            podMatrix[rowSize + 1] = 5;
            podMatrix[rowSize + 2] = 6;
            podMatrix[rowSize + 3] = 7;

            Console.WriteLine("Input plaintext matrix:");
            Utilities.PrintMatrix(podMatrix, (int)rowSize);
            Console.WriteLine();

            /*
            First we use BatchEncoder to encode the matrix into a plaintext. We encrypt
            the plaintext as usual.
            */
            Utilities.PrintLine();
            Plaintext plainMatrix = new Plaintext();
            Console.WriteLine("Encode and encrypt.");
            batchEncoder.Encode(podMatrix, plainMatrix);
            Ciphertext encryptedMatrix = new Ciphertext();
            encryptor.Encrypt(plainMatrix, encryptedMatrix);
            Console.WriteLine("    + Noise budget in fresh encryption: {0} bits",
                decryptor.InvariantNoiseBudget(encryptedMatrix));
            Console.WriteLine();

            /*
            Rotations require yet another type of special key called `Galois keys'. These
            are easily obtained from the KeyGenerator.旋转需要另一种特殊的密钥，称为“galoismi密钥”。
			这些很容易从密钥发生器获得。
            */
            GaloisKeys galKeys = keygen.GaloisKeys();

            /*
            Now rotate both matrix rows 3 steps to the left, decrypt, decode, and print.现在向左旋转矩阵行3步，解密、解码和打印。
            */
            Utilities.PrintLine();
            Console.WriteLine("Rotate rows 3 steps left.");
            evaluator.RotateRowsInplace(encryptedMatrix, 3, galKeys);
            Plaintext plainResult = new Plaintext();
            Console.WriteLine("    + Noise budget after rotation: {0} bits",
                decryptor.InvariantNoiseBudget(encryptedMatrix));
            Console.WriteLine("    + Decrypt and decode ...... Correct.");
            decryptor.Decrypt(encryptedMatrix, plainResult);
            List<ulong> podResult = new List<ulong>();
            batchEncoder.Decode(plainResult, podResult);
            Utilities.PrintMatrix(podResult, (int)rowSize);

            /*
            We can also rotate the columns, i.e., swap the rows.
            */
            Utilities.PrintLine();
            Console.WriteLine("Rotate columns.");
            evaluator.RotateColumnsInplace(encryptedMatrix, galKeys);
            Console.WriteLine("    + Noise budget after rotation: {0} bits",
                decryptor.InvariantNoiseBudget(encryptedMatrix));
            Console.WriteLine("    + Decrypt and decode ...... Correct.");
            decryptor.Decrypt(encryptedMatrix, plainResult);
            batchEncoder.Decode(plainResult, podResult);
            Utilities.PrintMatrix(podResult, (int)rowSize);

            /*
            Finally, we rotate the rows 4 steps to the right, decrypt, decode, and print.最后，我们向右旋转行4步，解密、解码和打印。
            */
            Utilities.PrintLine();
            Console.WriteLine("Rotate rows 4 steps right.");
            evaluator.RotateRowsInplace(encryptedMatrix, -4, galKeys);
            Console.WriteLine("    + Noise budget after rotation: {0} bits",
                decryptor.InvariantNoiseBudget(encryptedMatrix));
            Console.WriteLine("    + Decrypt and decode ...... Correct.");
            decryptor.Decrypt(encryptedMatrix, plainResult);
            batchEncoder.Decode(plainResult, podResult);
            Utilities.PrintMatrix(podResult, (int)rowSize);

            /*
            Note that rotations do not consume any noise budget. However, this is only
            the case when the special prime is at least as large as the other primes. The
            same holds for relinearization. Microsoft SEAL does not require that the
            special prime is of any particular size, so ensuring this is the case is left
            for the user to do.请注意，旋转不会消耗任何噪声预算。然而，这仅是当特殊素数至少与
			其它素数一样大的情况。用于再线性化的同样的保持。MicrosoftSeal不要求特殊的Prime具有任何特定的大小，
			因此确保为用户保留此情况。
            */
        }

        private static void ExampleRotationCKKS()
        {
            Utilities.PrintExampleBanner("Example: Rotation / Rotation in CKKS");

            EncryptionParameters parms = new EncryptionParameters(SchemeType.CKKS);

            ulong polyModulusDegree = 8192;
            parms.PolyModulusDegree = polyModulusDegree;
            parms.CoeffModulus = CoeffModulus.Create(
                polyModulusDegree, new int[] { 40, 40, 40, 40, 40 });

            SEALContext context = new SEALContext(parms);
            Utilities.PrintParameters(context);
            Console.WriteLine();

            KeyGenerator keygen = new KeyGenerator(context);
            PublicKey publicKey = keygen.PublicKey;
            SecretKey secretKey = keygen.SecretKey;
            RelinKeys relinKeys = keygen.RelinKeys();
            GaloisKeys galKeys = keygen.GaloisKeys();
            Encryptor encryptor = new Encryptor(context, publicKey);
            Evaluator evaluator = new Evaluator(context);
            Decryptor decryptor = new Decryptor(context, secretKey);

            CKKSEncoder ckksEncoder = new CKKSEncoder(context);

            ulong slotCount = ckksEncoder.SlotCount;
            Console.WriteLine($"Number of slots: {slotCount}");

            List<double> input = new List<double>((int)slotCount);
            double currPoint = 0, stepSize = 1.0 / (slotCount - 1);
            for (ulong i = 0; i < slotCount; i++, currPoint += stepSize)
            {
                input.Add(currPoint);
            }
            Console.WriteLine("Input vector:");
            Utilities.PrintVector(input, 3, 7);

            double scale = Math.Pow(2.0, 50);

            Utilities.PrintLine();
            Console.WriteLine("Encode and encrypt.");
            Plaintext plain = new Plaintext();
            ckksEncoder.Encode(input, scale, plain);
            Ciphertext encrypted = new Ciphertext();
            encryptor.Encrypt(plain, encrypted);

            Ciphertext rotated = new Ciphertext();
            Utilities.PrintLine();
            Console.WriteLine("Rotate 2 steps left.");
            evaluator.RotateVector(encrypted, 2, galKeys, rotated);
            Console.WriteLine("    + Decrypt and decode ...... Correct.");
            decryptor.Decrypt(encrypted, plain);
            List<double> result = new List<double>();
            ckksEncoder.Decode(plain, result);
            Utilities.PrintVector(result, 3, 7);

            /*
            With the CKKS scheme it is also possible to evaluate a complex conjugation on
            a vector of encrypted complex numbers, using Evaluator.ComplexConjugate. This
            is in fact a kind of rotation, and requires also Galois keys.
            */
        }

        private static void ExampleRotation()
        {
            Utilities.PrintExampleBanner("Example: Rotation");

            /*
            Run all rotation examples.
            */
            ExampleRotationBFV();
            ExampleRotationCKKS();
        }
    }
}
