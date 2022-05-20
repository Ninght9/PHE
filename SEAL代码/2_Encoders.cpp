    
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

using System;
using System.Collections.Generic;
using Microsoft.Research.SEAL;

namespace SEALNetExamples
{
    partial class Examples
    {
        /*
        In `1_BFV_Basics.cs' we showed how to perform a very simple computation using the
        BFV scheme. The computation was performed modulo the PlainModulus parameter, and
        utilized only one coefficient from a BFV plaintext polynomial. This approach has
        two notable problems:
		在“1_BfV_basics.cs”中，我们展示了如何使用BfV方案执行非常简单的计算。计算采用模块化的方法，
		仅使用BfV明文多项式中的一个系数。这种方法有两个值得注意的问题：
            (1) Practical applications typically use integer or real number arithmetic,
                not modular arithmetic;实际应用通常使用整数或实数算法，而不是模块算法；
            (2) We used only one coefficient of the plaintext polynomial. This is really
                wasteful, as the plaintext polynomial is large and will in any case be
                encrypted in its entirety.我们只使用明文多项式的一个系数。这是非常浪费的，因为明文多项式很大，无论如何都会被加密。
        For (1), one may ask why not just increase the PlainModulus parameter until no
        overflow occurs, and the computations behave as in integer arithmetic. The problem
        is that increasing PlainModulus increases noise budget consumption, and decreases
        the initial noise budget too.对于(1)，人们可能会问，为什么不只是增加平模参数，直到没有溢出发生，并且计算行为与整数算法一样。
		问题是，增加平模不仅增加了噪声预算消耗，而且降低了初始噪声预算。
        In these examples we will discuss other ways of laying out data into plaintext
        elements (encoding) that allow more computations without data type overflow, and
        can allow the full plaintext polynomial to be utilized.在这些示例中，我们将讨论将数据放置到明文元素(编码)中的其他方法，
		这些方法允许在不发生数据类型溢出的情况下进行更多的计算，并允许使用完整的明文多项式。
        */
        private static void ExampleIntegerEncoder()
        {
            Utilities.PrintExampleBanner("Example: Encoders / Integer Encoder");

            /*
            [IntegerEncoder] (For BFV scheme only)
            The IntegerEncoder encodes integers to BFV plaintext polynomials as follows.
            First, a binary expansion of the integer is computed. Next, a polynomial is
            created with the bits as coefficients. For example, the integer
			整数编码器将整数编码为BfV明文多项式，如下所示。首先，计算整数的二进制展开。然后，以位作为系数创建多项式。例如，整数
                26 = 2^4 + 2^3 + 2^1
            is encoded as the polynomial 1x^4 + 1x^3 + 1x^1. Conversely, plaintext
            polynomials are decoded by evaluating them at x=2. For negative numbers the
            IntegerEncoder simply stores all coefficients as either 0 or -1, where -1 is
            represented by the unsigned integer PlainModulus - 1 in memory.
            Since encrypted computations operate on the polynomials rather than on the
            encoded integers themselves, the polynomial coefficients will grow in the
            course of such computations. For example, computing the sum of the encrypted
            encoded integer 26 with itself will result in an encrypted polynomial with
            larger coefficients: 2x^4 + 2x^3 + 2x^1. Squaring the encrypted encoded
            integer 26 results also in increased coefficients due to cross-terms, namely,
			编码为多项式1x^4 1x^3 1x^1。相反，明文多项式在x=2处通过求它们来解码，对于负数，整数编码器只将所有系数存储为0或-1，其
			中-1由内存中的无符号整数平坦模-1表示。由于加密计算是在多项式上进行的，而不是对编码整数本身进行的，因此
			多项式系数在这种计算过程中会增长。例如，用它本身计算加密编码整数26的和将得到一个系数更大的加密多项式：2x^4 2x^3 2x^1。
                (1x^4 + 1x^3 + 1x^1)^2 = 1x^8 + 2x^7 + 1x^6 + 2x^5 + 2x^4 + 1x^2;
            further computations will quickly increase the coefficients much more.
            Decoding will still work correctly in this case (evaluating the polynomial
            at x=2), but since the coefficients of plaintext polynomials are really
            integers modulo plain_modulus, implicit reduction modulo plain_modulus may
            yield unexpected results. For example, adding 1x^4 + 1x^3 + 1x^1 to itself
            plain_modulus many times will result in the constant polynomial 0, which is
            clearly not equal to 26 * plain_modulus. It can be difficult to predict when
            such overflow will take place especially when computing several sequential
            multiplications.进一步的计算将很快地增加系数。在这种情况下，译码仍然是正确的(在x=2处计算多项式)，但是由于明文多项式的系数实
			际上是整数模素模，所以隐式缩减模素模可能会产生意想不到的结果。例如，将1x^4 1x^3 1x^1多次添加到其本身中，将导致常数多项式0，
			这显然不等于26*平面模。很难预测何时会发生这种溢出，特别是在计算几个顺序乘法时。
            The IntegerEncoder is easy to understand and use for simple computations,
            and can be a good tool to experiment with for users new to Microsoft SEAL.
            However, advanced users will probably prefer more efficient approaches,
            such as the BatchEncoder or the CKKSEncoder.
IntegerEncoder易于理解，可用于简单计算，并且可以作为Microsoft SEAL新用户进行实验的好工具，
但是，高级用户可能会喜欢更有效的方法，例如BatchEncoder或CKKSEncoder。
            */
            EncryptionParameters parms = new EncryptionParameters(SchemeType.BFV);
            ulong polyModulusDegree = 4096;
            parms.PolyModulusDegree = polyModulusDegree;
            parms.CoeffModulus = CoeffModulus.BFVDefault(polyModulusDegree);

            /*
            There is no hidden logic behind our choice of the plain_modulus. The only
            thing that matters is that the plaintext polynomial coefficients will not
            exceed this value at any point during our computation; otherwise the result
            will be incorrect.在我们选择素模的背后没有隐藏的逻辑。唯一重要的是，纯文本多项式系数在计算过程中的任何一点都不会超过这个值，否则结果将是不正确的。
            */
            parms.PlainModulus = new SmallModulus(512);
            SEALContext context = new SEALContext(parms);
            Utilities.PrintParameters(context);
            Console.WriteLine();

            KeyGenerator keygen = new KeyGenerator(context);
            PublicKey publicKey = keygen.PublicKey;
            SecretKey secretKey = keygen.SecretKey;
            Encryptor encryptor = new Encryptor(context, publicKey);
            Evaluator evaluator = new Evaluator(context);
            Decryptor decryptor = new Decryptor(context, secretKey);

            /*
            We create an IntegerEncoder.
            */
            IntegerEncoder encoder = new IntegerEncoder(context);

            /*
            First, we encode two integers as plaintext polynomials. Note that encoding
            is not encryption: at this point nothing is encrypted.首先，我们将两个整数编码为明文多项式。注意，编码不是加密：此时没有加密。
            */
            int value1 = 5;
            Plaintext plain1 = encoder.Encode(value1);
            Utilities.PrintLine();
            Console.WriteLine($"Encode {value1} as polynomial {plain1} (plain1),");

            int value2 = -7;
            Plaintext plain2 = encoder.Encode(value2);
            Console.WriteLine(new string(' ', 13)
                + $"Encode {value2} as polynomial {plain2} (plain2),");

            /*
            Now we can encrypt the plaintext polynomials.现在，我们可以加密明文多项式。
            */
            Ciphertext encrypted1 = new Ciphertext();
            Ciphertext encrypted2 = new Ciphertext();
            Utilities.PrintLine();
            Console.WriteLine("Encrypt plain1 to encrypted1 and plain2 to encrypted2.");
            encryptor.Encrypt(plain1, encrypted1);
            encryptor.Encrypt(plain2, encrypted2);
            Console.WriteLine("    + Noise budget in encrypted1: {0} bits",
                decryptor.InvariantNoiseBudget(encrypted1));
            Console.WriteLine("    + Noise budget in encrypted2: {0} bits",
                decryptor.InvariantNoiseBudget(encrypted2));

            /*
            As a simple example, we compute (-encrypted1 + encrypted2) * encrypted2.
            */
            encryptor.Encrypt(plain2, encrypted2);
            Ciphertext encryptedResult = new Ciphertext();
            Utilities.PrintLine();
            Console.WriteLine("Compute encrypted_result = (-encrypted1 + encrypted2) * encrypted2.");
            evaluator.Negate(encrypted1, encryptedResult);
            evaluator.AddInplace(encryptedResult, encrypted2);
            evaluator.MultiplyInplace(encryptedResult, encrypted2);
            Console.WriteLine("    + Noise budget in encryptedResult: {0} bits",
                decryptor.InvariantNoiseBudget(encryptedResult));

            Plaintext plainResult = new Plaintext();
            Utilities.PrintLine();
            Console.WriteLine("Decrypt encrypted_result to plain_result.");
            decryptor.Decrypt(encryptedResult, plainResult);

            /*
            Print the result plaintext polynomial. The coefficients are not even close
            to exceeding our plainModulus, 512.打印结果的明文多项式。这些系数甚至没有超过我们的标准模数512。
            */
            Console.WriteLine($"    + Plaintext polynomial: {plainResult}");

            /*
            Decode to obtain an integer result.
            */
            Utilities.PrintLine();
            Console.WriteLine("Decode plain_result.");
            Console.WriteLine("    + Decoded integer: {0} ...... Correct.",
                encoder.DecodeInt32(plainResult));
        }

        private static void ExampleBatchEncoder()
        {
            Utilities.PrintExampleBanner("Example: Encoders / Batch Encoder");

            /*
            [BatchEncoder] (For BFV scheme only)
            Let N denote the PolyModulusDegree and T denote the PlainModulus. Batching
            allows the BFV plaintext polynomials to be viewed as 2-by-(N/2) matrices, with
            each element an integer modulo T. In the matrix view, encrypted operations act
            element-wise on encrypted matrices, allowing the user to obtain speeds-ups of
            several orders of magnitude in fully vectorizable computations. Thus, in all
            but the simplest computations, batching should be the preferred method to use
            with BFV, and when used properly will result in implementations outperforming
            anything done with the IntegerEncoder.设n表示多模块度，t表示平坦度。批处理允许bfv明文
			多项式被看作2-by-(n/2)矩阵，每个元素是整数模t。在矩阵视图中，加密的操作在加密的矩阵上动作元素-wise，允许
			用户在完全可量化的计算中获得几个数量级的速度。因此，在最简单的计算中，批处理应该是与BFV一起使用的首选方法，
			并且当正确使用时，将导致实现与集成编码器执行的任何操作。
            */
            EncryptionParameters parms = new EncryptionParameters(SchemeType.BFV);
            ulong polyModulusDegree = 8192;
            parms.PolyModulusDegree = polyModulusDegree;
            parms.CoeffModulus = CoeffModulus.BFVDefault(polyModulusDegree);

            /*
            To enable batching, we need to set the plain_modulus to be a prime number
            congruent to 1 modulo 2*PolyModulusDegree. Microsoft SEAL provides a helper
            method for finding such a prime. In this example we create a 20-bit prime
            that supports batching.为了启用批处理，我们需要将平模设置为一个素数，同余为1模2*多模度。
			微软印章提供了一个帮助方法来找到这样的素数。在本例中，我们创建了一个支持批处理的20位素数。
            */
            parms.PlainModulus = PlainModulus.Batching(polyModulusDegree, 20);

            SEALContext context = new SEALContext(parms);
            Utilities.PrintParameters(context);
            Console.WriteLine();

            /*
            We can verify that batching is indeed enabled by looking at the encryption
            parameter qualifiers created by SEALContext.我们可以通过查看sealtext创建的加密参数限定符来验证批处理确实是启用的
            */
            var qualifiers = context.FirstContextData.Qualifiers;
            Console.WriteLine($"Batching enabled: {qualifiers.UsingBatching}");

            KeyGenerator keygen = new KeyGenerator(context);
            PublicKey publicKey = keygen.PublicKey;
            SecretKey secretKey = keygen.SecretKey;
            RelinKeys relinKeys = keygen.RelinKeys();
            Encryptor encryptor = new Encryptor(context, publicKey);
            Evaluator evaluator = new Evaluator(context);
            Decryptor decryptor = new Decryptor(context, secretKey);

            /*
            Batching is done through an instance of the BatchEncoder class.批处理是通过批编码类的一个实例来完成的。
            */
            BatchEncoder batchEncoder = new BatchEncoder(context);

            /*
            The total number of batching `slots' equals the PolyModulusDegree, N, and
            these slots are organized into 2-by-(N/2) matrices that can be encrypted and
            computed on. Each slot contains an integer modulo PlainModulus.批处理“时隙”的总数等于多模块度，n，并且这些时隙被组织成2到(n/2)矩阵，
			这些矩阵可以被加密和计算。每个时隙包含整数模数平坦模数。
            */
            ulong slotCount = batchEncoder.SlotCount;
            ulong rowSize = slotCount / 2;
            Console.WriteLine($"Plaintext matrix row size: {rowSize}");

            /*
            The matrix plaintext is simply given to BatchEncoder as a flattened vector
            of numbers. The first `rowSize' many numbers form the first row, and the
            rest form the second row. Here we create the following matrix:
                [ 0,  1,  2,  3,  0,  0, ...,  0 ]
                [ 4,  5,  6,  7,  0,  0, ...,  0 ]
            */
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

            /*
            First we use BatchEncoder to encode the matrix into a plaintext polynomial.首先，我们使用BatchEncoder将矩阵编码为明文多项式。
            */
            Plaintext plainMatrix = new Plaintext();
            Utilities.PrintLine();
            Console.WriteLine("Encode plaintext matrix:");
            batchEncoder.Encode(podMatrix, plainMatrix);

            /*
            We can instantly decode to verify correctness of the encoding. Note that no
            encryption or decryption has yet taken place.我们可以立即解码以验证编码的正确性。请注意，尚未进行加密或解密。
            */
            List<ulong> podResult = new List<ulong>();
            Console.WriteLine("    + Decode plaintext matrix ...... Correct.");
            batchEncoder.Decode(plainMatrix, podResult);
            Utilities.PrintMatrix(podResult, (int)rowSize);

            /*
            Next we encrypt the encoded plaintext.
            */
            Ciphertext encryptedMatrix = new Ciphertext();
            Utilities.PrintLine();
            Console.WriteLine("Encrypt plainMatrix to encryptedMatrix.");
            encryptor.Encrypt(plainMatrix, encryptedMatrix);
            Console.WriteLine("    + Noise budget in encryptedMatrix: {0} bits",
                decryptor.InvariantNoiseBudget(encryptedMatrix));

            /*
            Operating on the ciphertext results in homomorphic operations being performed
            simultaneously in all 8192 slots (matrix elements). To illustrate this, we
            form another plaintext matrix在密文上操作导致在所有8192个槽(矩阵元素)中同时执行同态操作。为了说明这一点，我们形成了另一个纯文本矩阵。
                [ 1,  2,  1,  2,  1,  2, ..., 2 ]
                [ 1,  2,  1,  2,  1,  2, ..., 2 ]
            and encode it into a plaintext.
            */
            ulong[] podMatrix2 = new ulong[slotCount];
            for (ulong i = 0; i < slotCount; i++)
            {
                podMatrix2[i] = (i % 2) + 1;
            }
            Plaintext plainMatrix2 = new Plaintext();
            batchEncoder.Encode(podMatrix2, plainMatrix2);
            Console.WriteLine();
            Console.WriteLine("Second input plaintext matrix:");
            Utilities.PrintMatrix(podMatrix2, (int)rowSize);

            /*
            We now add the second (plaintext) matrix to the encrypted matrix, and square
            the sum.我们现在将第二个(明文)矩阵添加到加密矩阵中，并将其平方。
            */
            Utilities.PrintLine();
            Console.WriteLine("Sum, square, and relinearize.");
            evaluator.AddPlainInplace(encryptedMatrix, plainMatrix2);
            evaluator.SquareInplace(encryptedMatrix);
            evaluator.RelinearizeInplace(encryptedMatrix, relinKeys);

            /*
            How much noise budget do we have left?
            */
            Console.WriteLine("    + Noise budget in result: {0} bits",
                decryptor.InvariantNoiseBudget(encryptedMatrix));

            /*
            We decrypt and decompose the plaintext to recover the result as a matrix.我们解密和分解明文以将结果恢复为矩阵。
            */
            Plaintext plainResult = new Plaintext();
            Utilities.PrintLine();
            Console.WriteLine("Decrypt and decode result.");
            decryptor.Decrypt(encryptedMatrix, plainResult);
            batchEncoder.Decode(plainResult, podResult);
            Console.WriteLine("    + Result plaintext matrix ...... Correct.");
            Utilities.PrintMatrix(podResult, (int)rowSize);

            /*
            Batching allows us to efficiently use the full plaintext polynomial when the
            desired encrypted computation is highly parallelizable. However, it has not
            solved the other problem mentioned in the beginning of this file: each slot
            holds only an integer modulo plain_modulus, and unless plain_modulus is very
            large, we can quickly encounter data type overflow and get unexpected results
            when integer computations are desired. Note that overflow cannot be detected
            in encrypted form. The CKKS scheme (and the CKKSEncoder) addresses the data
            type overflow issue, but at the cost of yielding only approximate results.当期望的加密计算高度并行时，
			批处理允许用户有效地使用完整的明文多项式。但是，它还没有解决本文件开头所述的另一个问题：每个插槽仅保存
			一个整数模数纯模数，除非纯模数非常大，否则我们可以快速遇到数据类型溢出，并在需要整数计算时获得意想不到的结果。
			请注意，无法以加密形式检测溢出。CKS方案（和CKKSEN编码器）处理数据类型溢出问题，但只产生近似结果的成本
            */
        }

        static private void ExampleCKKSEncoder()
        {
            Utilities.PrintExampleBanner("Example: Encoders / CKKS Encoder");

            /*
            [CKKSEncoder] (For CKKS scheme only)
            In this example we demonstrate the Cheon-Kim-Kim-Song (CKKS) scheme for
            computing on encrypted real or complex numbers. We start by creating
            encryption parameters for the CKKS scheme. There are two important
            differences compared to the BFV scheme:在本例中，我们演示了Cheon-Kim-Kim-Song(CKS)方案，用于计算加密的实数或复数。
			我们开始创建CKS方案的加密参数。与BFV方案相比，存在两个重要差异：
                (1) CKKS does not use the PlainModulus encryption parameter;
                (2) Selecting the CoeffModulus in a specific way can be very important
                    when using the CKKS scheme. We will explain this further in the file
                    `CKKS_Basics.cs'. In this example we use CoeffModulus.Create to
                    generate 5 40-bit prime numbers.
					(1)Cink不使用平坦模加密参数；(2)在使用CKS方案时，以一种特定的方式选择系数是非常重要的。我们将在文件‘ckks_basics.cs’中进一步解释这一点
					在本例中，我们使用coeffmous.create生成5个40位素数
            */
            EncryptionParameters parms = new EncryptionParameters(SchemeType.CKKS);

            ulong polyModulusDegree = 8192;
            parms.PolyModulusDegree = polyModulusDegree;
            parms.CoeffModulus = CoeffModulus.Create(
                polyModulusDegree, new int[]{ 40, 40, 40, 40, 40 });

            /*
            We create the SEALContext as usual and print the parameters.
            */
            SEALContext context = new SEALContext(parms);
            Utilities.PrintParameters(context);
            Console.WriteLine();

            /*
            Keys are created the same way as for the BFV scheme.
            */
            KeyGenerator keygen = new KeyGenerator(context);
            PublicKey publicKey = keygen.PublicKey;
            SecretKey secretKey = keygen.SecretKey;
            RelinKeys relinKeys = keygen.RelinKeys();

            /*
            We also set up an Encryptor, Evaluator, and Decryptor as usual.我们还像往常一样设置了一个加密器、评估器和解密器。
            */
            Encryptor encryptor = new Encryptor(context, publicKey);
            Evaluator evaluator = new Evaluator(context);
            Decryptor decryptor = new Decryptor(context, secretKey);

            /*
            To create CKKS plaintexts we need a special encoder: there is no other way
            to create them. The IntegerEncoder and BatchEncoder cannot be used with the
            CKKS scheme. The CKKSEncoder encodes vectors of real or complex numbers into
            Plaintext objects, which can subsequently be encrypted. At a high level this
            looks a lot like what BatchEncoder does for the BFV scheme, but the theory
            behind it is completely different.要创建手写文本，我们需要一个特殊的编码器：没有其他方法来创建它们。
			整数编码器和批编码器不能与INKKS方案一起使用。本发明将实数或复数的向量编码成明文对象，随后可对其进行加密。
			在较高的水平上，这看起来很像批编码器为BfV方案所做的，但背后的理论是完全不同的。
            */
            CKKSEncoder encoder = new CKKSEncoder(context);

            /*
            In CKKS the number of slots is PolyModulusDegree / 2 and each slot encodes
            one real or complex number. This should be contrasted with BatchEncoder in
            the BFV scheme, where the number of slots is equal to PolyModulusDegree
            and they are arranged into a matrix with two rows.在CKS中，时隙的数目是多模块度/2，
			并且每个时隙编码一个实数或复数。这应该与BFV方案中的BatchEncoder进行对比，其中时隙的数量等于多模块度，
			并且它们被布置成具有两行的矩阵。
            */
            ulong slotCount = encoder.SlotCount;
            Console.WriteLine($"Number of slots: {slotCount}");

            /*
            We create a small vector to encode; the CKKSEncoder will implicitly pad it
            with zeros to full size (PolyModulusDegree / 2) when encoding.我们创建一个用于编码的小向量；当编码时，
			ckksencoder将隐式地将其嵌入零到完全大小(多模度/2)。
            */
            double[] input = new double[]{ 0.0, 1.1, 2.2, 3.3 };
            Console.WriteLine("Input vector: ");
            Utilities.PrintVector(input);

            /*
            Now we encode it with CKKSEncoder. The floating-point coefficients of `input'
            will be scaled up by the parameter `scale'. This is necessary since even in
            the CKKS scheme the plaintext elements are fundamentally polynomials with
            integer coefficients. It is instructive to think of the scale as determining
            the bit-precision of the encoding; naturally it will affect the precision of
            the result.现在我们用ckksencoder对它进行编码。“输入”的浮点系数将按参数“比例”放大。这是必要的，因为即使在CKS方案中，
			明文元素基本上是具有整数系数的多项式。把尺度看作确定编码的比特精度是有益的；当然，它将影响结果的精度
            In CKKS the message is stored modulo CoeffModulus (in BFV it is stored modulo
            PlainModulus), so the scaled message must not get too close to the total size
            of CoeffModulus. In this case our CoeffModulus is quite large (200 bits) so
            we have little to worry about in this regard. For this simple example a 30-bit
            scale is more than enough.在WINKS中，消息被存储为模系数(在BfV中它是存储模素模)，因此缩放消息不能太接近系数的总大小。在这种情况下，我们的系数很大(200位)，
			所以我们在这方面没有什么可担心的。对于这个简单的例子，一个30位的比例尺就足够了。
            */
            Plaintext plain = new Plaintext();
            double scale = Math.Pow(2.0, 30);
            Utilities.PrintLine();
            Console.WriteLine("Encode input vector.");
            encoder.Encode(input, scale, plain);

            /*
            We can instantly decode to check the correctness of encoding.
            */
            List<double> output = new List<double>();
            Console.WriteLine("    + Decode input vector ...... Correct.");
            encoder.Decode(plain, output);
            Utilities.PrintVector(output);

            /*
            The vector is encrypted the same was as in BFV.
            */
            Ciphertext encrypted = new Ciphertext();
            Utilities.PrintLine();
            Console.WriteLine("Encrypt input vector, square, and relinearize.");
            encryptor.Encrypt(plain, encrypted);

            /*
            Basic operations on the ciphertexts are still easy to do. Here we square
            the ciphertext, decrypt, decode, and print the result. We note also that
            decoding returns a vector of full size (PolyModulusDegree / 2); this is
            because of the implicit zero-padding mentioned above.对密文的基本操作仍然很容易。在这里，我们对密文进行平方，解密，解码，
			并打印结果。我们还注意到，解码返回一个完全大小的向量(多模度/2)；这是因为上面提到的隐式零填充。
            */
            evaluator.SquareInplace(encrypted);
            evaluator.RelinearizeInplace(encrypted, relinKeys);

            /*
            We notice that the scale in the result has increased. In fact, it is now
            the square of the original scale: 2^60.我们注意到，结果中的比例已经增加。事实上，它现在是原始规模的平方：2。60。
            */
            Console.WriteLine("    + Scale in squared input: {0} ({1} bits)",
                encrypted.Scale,
                (int)Math.Ceiling(Math.Log(encrypted.Scale, newBase: 2)));
            Utilities.PrintLine();
            Console.WriteLine("Decrypt and decode.");
            decryptor.Decrypt(encrypted, plain);
            encoder.Decode(plain, output);
            Console.WriteLine("    + Result vector ...... Correct.");
            Utilities.PrintVector(output);

            /*
            The CKKS scheme allows the scale to be reduced between encrypted computations.
            This is a fundamental and critical feature that makes CKKS very powerful and
            flexible. We will discuss it in great detail in `3_Levels.cs' and later in
            `4_CKKS_Basics.cs'.该方案允许在加密计算之间缩小规模。这是一个基本的和关键的特点，
			使乳汁非常强大和灵活。我们将在‘3_level s.cs’和后面的‘4_ckks_basics.cs’中详细讨论它。
            */
        }

        private static void ExampleEncoders()
        {
            Utilities.PrintExampleBanner("Example: Encoders");

            /*
            Run all encoder examples.
            */
            ExampleIntegerEncoder();
            ExampleBatchEncoder();
            ExampleCKKSEncoder();
        }
    }
}
© 2019 GitHub, Inc.