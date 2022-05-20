using System;
using System.Collections.Generic;
using Microsoft.Research.SEAL;

namespace SEALNetExamples
{
    partial class Examples
    {
        private static void ExampleCKKSBasics()
        {
            Utilities.PrintExampleBanner("Example: CKKS Basics");

            /*
            In this example we demonstrate evaluating a polynomial function在这个例子中，我们演示了求多项式函数的方法。
                PI*x^3 + 0.4*x + 1
            on encrypted floating-point input data x for a set of 4096 equidistant points
            in the interval [0, 1]. This example demonstrates many of the main features
            of the CKKS scheme, but also the challenges in using it.关于间隔[0,1]中的一组4096个
			等距点的加密浮点输入数据X。
			
		此示例演示了CKS方案的许多主要功能，但也说明了使用它的挑战。
            We start by setting up the CKKS scheme.
            */
            EncryptionParameters parms = new EncryptionParameters(SchemeType.CKKS);

            /*
            We saw in `2_Encoders.cs' that multiplication in CKKS causes scales in
            ciphertexts to grow. The scale of any ciphertext must not get too close to
            the total size of CoeffModulus, or else the ciphertext simply runs out of
            room to store the scaled-up plaintext. The CKKS scheme provides a `rescale'
            functionality that can reduce the scale, and stabilize the scale expansion.
            Rescaling is a kind of modulus switch operation (recall `3_Levels.cs').
            As modulus switching, it removes the last of the primes from CoeffModulus,
            but as a side-effect it scales down the ciphertext by the removed prime.
			我们在“2_encoders.cs”中看到，CKS中的乘法导致密文中的比例增长。任何密文的比例
			不得过接近于有效模数的总大小，否则密文只跑出空间以存储经缩放的明文。CKS方案提
			供了一种“重新缩放”功能，可降低规模，并稳定规模扩展。Rescaling是一种模数开关操
			作（召回“3_levels.cs”）。作为模量切换，它从系数中去除最后一个素数，但是作为一
			个副作用，它通过去除的素数来缩小密文。
            Usually we want to have perfect control over how the scales are changed,
            which is why for the CKKS scheme it is more common to use carefully selected
            primes for the CoeffModulus.通常，我们希望对标度的变化有完美的控制，这就是为什么
			对于ckks方案来说，使用精心选择的素数来表示系数是比较普遍的。
            More precisely, suppose that the scale in a CKKS ciphertext is S, and the
            last prime in the current CoeffModulus (for the ciphertext) is P. Rescaling
            to the next level changes the scale to S/P, and removes the prime P from the
            CoeffModulus, as usual in modulus switching. The number of primes limits
            how many rescalings can be done, and thus limits the multiplicative depth of
            the computation.更确切地说，假设CKS密文中的比例是S，而当前系数(密文)中的最后一次是P。
			重新计算到下一级别将缩放为S/P，并从系数转换中移除PRIMEP(如通常在模数切换中)。素数的数目
			限制了可以如何进行多少重新缩放，并且因此限制了计算的乘法深度。
            It is possible to choose the initial scale freely. One good strategy can be
            to is to set the initial scale S and primes P_i in the CoeffModulus to be
            very close to each other. If ciphertexts have scale S before multiplication,
            they have scale S^2 after multiplication, and S^2/P_i after rescaling. If all
            P_i are close to S, then S^2/P_i is close to S again. This way we stabilize the
            scales to be close to S throughout the computation. Generally, for a circuit
            of depth D, we need to rescale D times, i.e., we need to be able to remove D
            primes from the coefficient modulus. Once we have only one prime left in the
            coeff_modulus, the remaining prime must be larger than S by a few bits to
            preserve the pre-decimal-point value of the plaintext.
            Therefore, a generally good strategy is to choose parameters for the CKKS
            scheme as follows:可以自由选择初始标度。一种很好的策略是将系数模中的初始标度s和素数p_i
			设置得非常接近。如果密文在乘法前有标度s，则在乘法后有s^2，重标度后有s^2/p_i。如果所有p_i都接近s，
			则s^2/p_i再次接近s。通过这种方法，我们可以在整个计算过程中将标度稳定到接近s。一般情况下，对于深度
			d的电路，我们需要重新计算d次，也就是说，我们需要能够从系数模中删除d素数。一旦我们在Coeff模中只剩下
			一个素数，剩下的素数必须比s大几位
                (1) Choose a 60-bit prime as the first prime in CoeffModulus. This will
                    give the highest precision when decrypting;
                (2) Choose another 60-bit prime as the last element of CoeffModulus, as
                    this will be used as the special prime and should be as large as the
                    largest of the other primes;
                (3) Choose the intermediate primes to be close to each other.
				(1)选择一个60位素数作为系数的第一个密文模数。这将在解密时提供最高的精度；
				(2)选择另一个60位素数作为密文模数的最后一个元素，因为这将用作特殊素数，并且应该与其
				他素数中的最大素数一样大；
				(3)选择中间素数彼此接近。
            We use CoeffModulus.Create to generate primes of the appropriate size. Note
            that our CoeffModulus is 200 bits total, which is below the bound for our
            PolyModulusDegree: CoeffModulus.MaxBitCount(8192) returns 218.
			我们使用coeffmodulus.create生成适当大小的素数。请注意，我们的Coefficient是200位，
			低于我们的PolyModulus学位的绑定：CoeffectModulusMaxBitCount(8192)返回218。
            */
            ulong polyModulusDegree = 8192;
            parms.PolyModulusDegree = polyModulusDegree;
            parms.CoeffModulus = CoeffModulus.Create(
                polyModulusDegree, new int[]{ 60, 40, 40, 60 });

            /*
            We choose the initial scale to be 2^40. At the last level, this leaves us
            60-40=20 bits of precision before the decimal point, and enough (roughly
            10-20 bits) of precision after the decimal point. Since our intermediate
            primes are 40 bits (in fact, they are very close to 2^40), we can achieve
            scale stabilization as described above.我们选择初始比例为2,40。在最后一级，这使得在小数点前的精
			度为60-40=20位，小数点后精度足够（约为10-20位）。由于我们的中间素数是40比特(实际上，它们非常接近2，40)，
			所以我们可以实现如上所述的规模稳定。
            */
            double scale = Math.Pow(2.0, 40);

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

            CKKSEncoder encoder = new CKKSEncoder(context);
            ulong slotCount = encoder.SlotCount;
            Console.WriteLine($"Number of slots: {slotCount}");

            List<double> input = new List<double>((int)slotCount);
            double currPoint = 0, stepSize = 1.0 / (slotCount - 1);
            for (ulong i = 0; i < slotCount; i++, currPoint += stepSize)
            {
                input.Add(currPoint);
            }
            Console.WriteLine("Input vector:");
            Utilities.PrintVector(input, 3, 7);

            Console.WriteLine("Evaluating polynomial PI*x^3 + 0.4x + 1 ...");

            /*
            We create plaintexts for PI, 0.4, and 1 using an overload of CKKSEncoder.Encode
            that encodes the given floating-point value to every slot in the vector.
            */
            Plaintext plainCoeff3 = new Plaintext(),
                      plainCoeff1 = new Plaintext(),
                      plainCoeff0 = new Plaintext();
            encoder.Encode(3.14159265, scale, plainCoeff3);
            encoder.Encode(0.4, scale, plainCoeff1);
            encoder.Encode(1.0, scale, plainCoeff0);

            Plaintext xPlain = new Plaintext();
            Utilities.PrintLine();
            Console.WriteLine("Encode input vectors.");
            encoder.Encode(input, scale, xPlain);
            Ciphertext x1Encrypted = new Ciphertext();
            encryptor.Encrypt(xPlain, x1Encrypted);

            /*
            To compute x^3 we first compute x^2 and relinearize. However, the scale has
            now grown to 2^80.为了计算x^3，我们首先计算x^2并重新定义。然而，现在的规模已扩大到2^80。
            */
            Ciphertext x3Encrypted = new Ciphertext();
            Utilities.PrintLine();
            Console.WriteLine("Compute x^2 and relinearize:");
            evaluator.Square(x1Encrypted, x3Encrypted);
            evaluator.RelinearizeInplace(x3Encrypted, relinKeys);
            Console.WriteLine("    + Scale of x^2 before rescale: {0} bits",
                Math.Log(x3Encrypted.Scale, newBase: 2));

            /*
            Now rescale; in addition to a modulus switch, the scale is reduced down by
            a factor equal to the prime that was switched away (40-bit prime). Hence, the
            new scale should be close to 2^40. Note, however, that the scale is not equal
            to 2^40: this is because the 40-bit prime is only close to 2^40.现在重新缩放；除了模数开关之外，
			该比例被降低了等于被切换的素数（40位素数）的因子。
			因此，新规模应接近2^40。然而，注意，该比例不等于2^40：这是因为40位的素数仅接近2。40。
            */
            Utilities.PrintLine();
            Console.WriteLine("Rescale x^2.");
            evaluator.RescaleToNextInplace(x3Encrypted);
            Console.WriteLine("    + Scale of x^2 after rescale: {0} bits",
                Math.Log(x3Encrypted.Scale, newBase: 2));

            /*
            Now x3Encrypted is at a different level than x1Encrypted, which prevents us
            from multiplying them to compute x^3. We could simply switch x1Encrypted to
            the next parameters in the modulus switching chain. However, since we still
            need to multiply the x^3 term with PI (plainCoeff3), we instead compute PI*x
            first and multiply that with x^2 to obtain PI*x^3. To this end, we compute
            PI*x and rescale it back from scale 2^80 to something close to 2^40.现在，x3加密与x1加密的级别不同，
			这使得我们无法将它们乘以计算x^3。我们可以简单地将x1加密切换到模数交换链中的下一个参数。但是，
			由于我们仍然需要用pi(平原系数3)乘以x^3项，
			所以我们首先计算pi*x，然后用x^2乘以它来得到pi*x^3。为此，我们计算pi*x，并将它从标度2
			^80恢复到接近2^40的值。
            */
            Utilities.PrintLine();
            Console.WriteLine("Compute and rescale PI*x.");
            Ciphertext x1EncryptedCoeff3 = new Ciphertext();
            evaluator.MultiplyPlain(x1Encrypted, plainCoeff3, x1EncryptedCoeff3);
            Console.WriteLine("    + Scale of PI*x before rescale: {0} bits",
                Math.Log(x1EncryptedCoeff3.Scale, newBase: 2));
            evaluator.RescaleToNextInplace(x1EncryptedCoeff3);
            Console.WriteLine("    + Scale of PI*x after rescale: {0} bits",
                Math.Log(x1EncryptedCoeff3.Scale, newBase: 2));

            /*
            Since x3Encrypted and x1EncryptedCoeff3 have the same exact scale and use
            the same encryption parameters, we can multiply them together. We write the
            result to x3Encrypted, relinearize, and rescale. Note that again the scale
            is something close to 2^40, but not exactly 2^40 due to yet another scaling
            by a prime. We are down to the last level in the modulus switching chain.
			现在，x3加密与x1加密的级别不同，这使得我们无法将它们乘以计算x^3。我们可以简单
			地将x1加密切换到模数交换链中的下一个参数。但是，由于我们仍然需要用pi(平原系数3)
			乘以x^3项，所以我们首先计算pi*x，然后用x^2乘以它来得到pi*x^3。为此，我们计算pi*x，
			并将它从标度2^80恢复到接近2^40的值。
            */
            Utilities.PrintLine();
            Console.WriteLine("Compute, relinearize, and rescale (PI*x)*x^2.");
            evaluator.MultiplyInplace(x3Encrypted, x1EncryptedCoeff3);
            evaluator.RelinearizeInplace(x3Encrypted, relinKeys);
            Console.WriteLine("    + Scale of PI*x^3 before rescale: {0} bits",
                Math.Log(x3Encrypted.Scale, newBase: 2));
            evaluator.RescaleToNextInplace(x3Encrypted);
            Console.WriteLine("    + Scale of PI*x^3 after rescale: {0} bits",
                Math.Log(x3Encrypted.Scale, newBase: 2));

            /*
            Next we compute the degree one term. All this requires is one MultiplyPlain
            with plainCoeff1. We overwrite x1Encrypted with the result.接下来，我们计算度一项。所有这一切都需要一
			个带有明文1的乘法器。我们用结果覆盖x1加密。
            */
            Utilities.PrintLine();
            Console.WriteLine("Compute and rescale 0.4*x.");
            evaluator.MultiplyPlainInplace(x1Encrypted, plainCoeff1);
            Console.WriteLine("    + Scale of 0.4*x before rescale: {0} bits",
                Math.Log(x1Encrypted.Scale, newBase: 2));
            evaluator.RescaleToNextInplace(x1Encrypted);
            Console.WriteLine("    + Scale of 0.4*x after rescale: {0} bits",
                Math.Log(x1Encrypted.Scale, newBase: 2));

            /*
            Now we would hope to compute the sum of all three terms. However, there is
            a serious problem: the encryption parameters used by all three terms are
            different due to modulus switching from rescaling.现在，我们希望计算所有三个项的总和。
			然而，也存在着一个严重的问题：由于模数的切换，所有三个项所使用的加密参数都是不同的。
            Encrypted addition and subtraction require that the scales of the inputs are
            the same, and also that the encryption parameters (ParmsId) match. If there
            is a mismatch, Evaluator will throw an exception.加密加减法要求输入的尺度相同，
			加密参数（parmsid）匹配。如果存在不匹配，评估器将抛出异常。
            */
            Console.WriteLine();
            Utilities.PrintLine();
            Console.WriteLine("Parameters used by all three terms are different:");
            Console.WriteLine("    + Modulus chain index for x3Encrypted: {0}",
                context.GetContextData(x3Encrypted.ParmsId).ChainIndex);
            Console.WriteLine("    + Modulus chain index for x1Encrypted: {0}",
                context.GetContextData(x1Encrypted.ParmsId).ChainIndex);
            Console.WriteLine("    + Modulus chain index for plainCoeff0: {0}",
                context.GetContextData(plainCoeff0.ParmsId).ChainIndex);
            Console.WriteLine();

            /*
            Let us carefully consider what the scales are at this point. We denote the
            primes in coeff_modulus as P_0, P_1, P_2, P_3, in this order. P_3 is used as
            the special modulus and is not involved in rescalings. After the computations
            above the scales in ciphertexts are:让我们仔细考虑在这一点上的规模是什么。以Coeff_模数为P_0、P_1、P_2、P_3表示素数。P_3用作特殊模量，
			并不参与重新试验。在上述计算之后，密文中的比例为：
                - Product x^2 has scale 2^80 and is at level 2;
                - Product PI*x has scale 2^80 and is at level 2;
                - We rescaled both down to scale 2^80/P2 and level 1;
                - Product PI*x^3 has scale (2^80/P_2)^2;
                - We rescaled it down to scale (2^80/P_2)^2/P_1 and level 0;
                - Product 0.4*x has scale 2^80;
                - We rescaled it down to scale 2^80/P_2 and level 1;
                - The contant term 1 has scale 2^40 and is at level 2.
            Although the scales of all three terms are approximately 2^40, their exact
            values are different, hence they cannot be added together.虽然这三个术语的标度大约是2^40，但它们的精确值是不同的，因此它们不能相加。
            */
            Utilities.PrintLine();
            Console.WriteLine("The exact scales of all three terms are different:");
            Console.WriteLine("    + Exact scale in PI*x^3: {0:0.0000000000}", x3Encrypted.Scale);
            Console.WriteLine("    + Exact scale in  0.4*x: {0:0.0000000000}", x1Encrypted.Scale);
            Console.WriteLine("    + Exact scale in      1: {0:0.0000000000}", plainCoeff0.Scale);
            Console.WriteLine();

            /*
            There are many ways to fix this problem. Since P_2 and P_1 are really close
            to 2^40, we can simply "lie" to Microsoft SEAL and set the scales to be the
            same. For example, changing the scale of PI*x^3 to 2^40 simply means that we
            scale the value of PI*x^3 by 2^120/(P_2^2*P_1), which is very close to 1.
            This should not result in any noticeable error.
			有很多方法可以解决这个问题。由于p_2和p_1实际上接近2^40，我们可以简单地“撒谎”给微软印章，
			并将天平设置为相同。例如，将pi*x^3的标度更改为2^40只意味着我们将pi*x^3的值缩放为2^120/(p_2^2*p_1)，
			这非常接近于1。这不应导致任何明显的错误。
            Another option would be to encode 1 with scale 2^80/P_2, do a MultiplyPlain
            with 0.4*x, and finally rescale. In this case we would need to additionally
            make sure to encode 1 with appropriate encryption parameters (ParmsId).
            In this example we will use the first (simplest) approach and simply change
            the scale of PI*x^3 and 0.4*x to 2^40.另一种选择是用2,80/p_2的比例对1进行编码,
			用0.4*x进行乘法,最后重新缩放。在这种情况下，我们需要额外地确保用适当的加密参数(parmsid)来编码1。
			在本例中，我们将使用第一个（最简单的）方法，并简单地将pi*x、3和0.4*x的比例更改为2。40。
            */
            Utilities.PrintLine();
            Console.WriteLine("Normalize scales to 2^40.");
            x3Encrypted.Scale = Math.Pow(2.0, 40);
            x1Encrypted.Scale = Math.Pow(2.0, 40);

            /*
            We still have a problem with mismatching encryption parameters. This is easy
            to fix by using traditional modulus switching (no rescaling). CKKS supports
            modulus switching just like the BFV scheme, allowing us to switch away parts
            of the coefficient modulus when it is simply not needed.286/5000
加密参数不匹配仍然存在问题。 通过使用传统的模数转换（无需重新缩放），可以轻松解决此问题。 
就像BFV方案一样，CKKS支持模数转换，从而使我们在根本不需要时就可以将系数模数的部分移开。
            */
            Utilities.PrintLine();
            Console.WriteLine("Normalize encryption parameters to the lowest level.");
            ParmsId lastParmsId = x3Encrypted.ParmsId;
            evaluator.ModSwitchToInplace(x1Encrypted, lastParmsId);
            evaluator.ModSwitchToInplace(plainCoeff0, lastParmsId);

            /*
            All three ciphertexts are now compatible and can be added.现在，所有三个密文都兼容并且可以添加。
            */
            Utilities.PrintLine();
            Console.WriteLine("Compute PI*x^3 + 0.4*x + 1.");
            Ciphertext encryptedResult = new Ciphertext();
            evaluator.Add(x3Encrypted, x1Encrypted, encryptedResult);
            evaluator.AddPlainInplace(encryptedResult, plainCoeff0);

            /*
            First print the true result.
            */
            Plaintext plainResult = new Plaintext();
            Utilities.PrintLine();
            Console.WriteLine("Decrypt and decode PI * x ^ 3 + 0.4x + 1.");
            Console.WriteLine("    + Expected result:");
            List<double> trueResult = new List<double>(input.Count);
            foreach (double x in input)
            {
                trueResult.Add((3.14159265 * x * x + 0.4) * x + 1);
            }
            Utilities.PrintVector(trueResult, 3, 7);

            /*
            We decrypt, decode, and print the result.
            */
            decryptor.Decrypt(encryptedResult, plainResult);
            List<double> result = new List<double>();
            encoder.Decode(plainResult, result);
            Console.WriteLine("    + Computed result ...... Correct.");
            Utilities.PrintVector(result, 3, 7);

            /*
            While we did not show any computations on complex numbers in these examples,
            the CKKSEncoder would allow us to have done that just as easily. Additions
            and multiplications of complex numbers behave just as one would expect.虽然在这些例子中，我们没有对复杂数字进行任何计算，
			但是CKKSENSE编码器将允许我们做到这一点很容易。复杂数字的添加和乘法行为正如人们所期望的一样。
            */
        }
    }
}