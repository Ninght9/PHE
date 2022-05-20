using System;
using Microsoft.Research.SEAL;

namespace SEALNetExamples
{
    partial class Examples
    {
        private static void ExampleBFVBasics()
        {
            Utilities.PrintExampleBanner("Example: BFV Basics");

            /*
            In this example, we demonstrate performing simple computations (a polynomial  在此示例中，我们演示了使用BFV加密方案对加密的整数执行简单的计算（多项式求值）。第一个任务是设置EncryptionParameters类的实例。 了解不同参数的行为，它们如何影响加密方案，性能和安全级别至关重要。 需要设置三个加密参数：
            evaluation) on encrypted integers using the BFV encryption scheme.
            The first task is to set up an instance of the EncryptionParameters class.
            It is critical to understand how the different parameters behave, how they
            affect the encryption scheme, performance, and the security level. There are
            three encryption parameters that are necessary to set:
                - PolyModulusDegree (degree of polynomial modulus); 多项式模型的度（多项式的最高次数）
                - CoeffModulus ([ciphertext] coefficient modulus);    密文系数模数
                - PlainModulus (plaintext modulus; only for the BFV scheme).  明文模数
            The BFV scheme cannot perform arbitrary computations on encrypted data.   BFV方案无法对加密数据执行任意计算，而是每个密文都有一个特定的量，称为``不变噪声预算''（简称为``噪声预算''），以比特为单位。刚加密的密文中的噪声预算（初始噪声预算）由加密参数确定。同态运算以同样由加密参数确定的速率消耗噪声预算。
            Instead, each ciphertext has a specific quantity called the `invariant noise  在BFV中，加密数据允许的两个基本操作是加法和乘法，
            budget' -- or `noise budget' for short -- measured in bits. The noise budget  与乘法相比，就噪声预算消耗而言，通常可以认为加法几乎是免费的。
            in a freshly encrypted ciphertext (initial noise budget) is determined by  由于噪声预算消耗会按顺序相乘，因此选择合适的加密参数时，
            the encryption parameters. Homomorphic operations consume the noise budget  最重要的因素是用户要对加密数据进行评估的算术电路的相乘深度。
            at a rate also determined by the encryption parameters. In BFV the two basic 一旦密文的噪声预算达到零，它就会变得太损坏而无法解密。
            operations allowed on encrypted data are additions and multiplications, of   因此，至关重要的是选择足够大的参数以支持所需的计算。否则即使使用密钥也无法弄清结果
            which additions can generally be thought of as being nearly free in terms of
            noise budget consumption compared to multiplications. Since noise budget
            consumption compounds in sequential multiplications, the most significant
            factor in choosing appropriate encryption parameters is the multiplicative
            depth of the arithmetic circuit that the user wants to evaluate on encrypted
            data. Once the noise budget of a ciphertext reaches zero it becomes too
            corrupted to be decrypted. Thus, it is essential to choose the parameters to
            be large enough to support the desired computation; otherwise the result is
            impossible to make sense of even with the secret key.
            */
            EncryptionParameters parms = new EncryptionParameters(SchemeType.BFV);

            /*
            The first parameter we set is the degree of the `polynomial modulus'. This 我们设置的第一个参数是“多项式模量”的程度。 这必须是2的正乘方，代表2幂乘方的多项式的次数； 不必了解这意味着什么。
            must be a positive power of 2, representing the degree of a power-of-two   较大的PolyModulusDegree会使密文大小变大，并且所有操作都变慢，但启用更复杂的加密计算。
            cyclotomic polynomial; it is not necessary to understand what this means.   推荐值为1024、2048、4096、8192、16384、32768，但也可以超出此范围。 在此示例中，我们使用相对较小的多项式模量。
            Larger PolyModulusDegree makes ciphertext sizes larger and all operations   小于此值的任何内容将仅允许非常严格的加密计算。
            slower, but enables more complicated encrypted computations. Recommended
            values are 1024, 2048, 4096, 8192, 16384, 32768, but it is also possible
            to go beyond this range.
            In this example we use a relatively small polynomial modulus. Anything
            smaller than this will enable only very restricted encrypted computations.
            */
            ulong polyModulusDegree = 4096;
            parms.PolyModulusDegree = polyModulusDegree;

            /*
            Next we set the [ciphertext] `coefficient modulus' (CoeffModulus). This    接下来，我们设置[密文]“系数模量”（CoeffModulus）。 此参数是一个大整数，
            parameter is a large integer, which is a product of distinct prime numbers,  它是不同质数和数字的乘积，每个质数均由SmallModulus类的实例表示。 CoeffModulus的位长是指其主要因子的位长之和。
            numbers, each represented by an instance of the SmallModulus class. The
            bit-length of CoeffModulus means the sum of the bit-lengths of its prime
            factors.
            A larger CoeffModulus implies a larger noise budget, hence more encrypted  较大的CoeffModulus意味着较大的噪声预算，因此具有更多的加密计算能力
            computation capabilities. However, an upper bound for the total bit-length   但是，CoeffModulus的总位长的上限由PolyModulusDegree确定，如下所示：
            of the CoeffModulus is determined by the PolyModulusDegree, as follows:
                +----------------------------------------------------+
                | PolyModulusDegree   | max CoeffModulus bit-length  |
                +---------------------+------------------------------+
                | 1024                | 27                           |
                | 2048                | 54                           |
                | 4096                | 109                          |
                | 8192                | 218                          |
                | 16384               | 438                          |
                | 32768               | 881                          |
                +---------------------+------------------------------+
            These numbers can also be found in native/src/seal/util/hestdparms.h encoded 这些数字也可以在SEAL_HE_STD_PARMS_128_TC函数中编码的native / src / seal / util / hestdparms.h中找到，也可以从函数中获取
            in the function SEAL_HE_STD_PARMS_128_TC, and can also be obtained from the
            function
                CoeffModulus.MaxBitCount(polyModulusDegree).
            For example, if PolyModulusDegree is 4096, the coeff_modulus could consist
            of three 36-bit primes (108 bits).例如，如果PolyModulusDegree为4096，则coeff_modulus可以由三个36位素数（108位）组成。
            Microsoft SEAL comes with helper functions for selecting the CoeffModulus.
            For new users the easiest way is to simply use Microsoft SEAL带有用于选择CoeffModulus的帮助程序功能。对于新用户，最简单的方法是简单地使用
                CoeffModulus.BFVDefault(polyModulusDegree),
            which returns IEnumerable<SmallModulus> consisting of a generally good choice
            for the given PolyModulusDegree.
            */
            parms.CoeffModulus = CoeffModulus.BFVDefault(polyModulusDegree);

            /*
            The plaintext modulus can be any positive integer, even though here we take明文模数可以是任何正整数，即使在这里我们将其取为2的幂。 实际上，在许多情况下，人们可能希望它成为素数。
            it to be a power of two. In fact, in many cases one might instead want it
            to be a prime number; we will see this in later examples. The plaintext  我们将在后面的示例中看到这一点。 
            modulus determines the size of the plaintext data type and the consumption
            of noise budget in multiplications. Thus, it is essential to try to keep the 明文模数决定了明文数据类型的大小以及乘法运算中噪声预算的消耗。 因此，为保持最佳性能，必须设法使明文数据类型尽可能小。 新加密的密文中的噪声预算为
            plaintext data type as small as possible for best performance. The noise
            budget in a freshly encrypted ciphertext is
                ~ log2(CoeffModulus/PlainModulus) (bits)   噪声预算
            and the noise budget consumption in a homomorphic multiplication is of the
            form log2(PlainModulus) + (other terms).
            The plaintext modulus is specific to the BFV scheme, and cannot be set when
            using the CKKS scheme.并且同态乘法的噪声预算消耗的形式为log2（PlainModulus）+（其他术语）。明文模量特定于BFV方案，在设置时不能设置
使用CKKS方案。
            */
            parms.PlainModulus = new SmallModulus(1024);

            /*
            Now that all parameters are set, we are ready to construct a SEALContext
            object. This is a heavy class that checks the validity and properties of the
            parameters we just set. 
			现在已经设置了所有参数，我们准备构造SEALContext对象。 这是一个繁重的类，用于检查我们刚刚设置的参数的有效性和属性。
            */
            SEALContext context = new SEALContext(parms);

            /*
            Print the parameters that we have chosen.
            */
            Utilities.PrintLine();
            Console.WriteLine("Set encryption parameters and print");
            Utilities.PrintParameters(context);

            Console.WriteLine();
            Console.WriteLine("~~~~~~ A naive way to calculate 4(x^2+1)(x+1)^2. ~~~~~~");

            /*
            The encryption schemes in Microsoft SEAL are public key encryption schemes.
            For users unfamiliar with this terminology, a public key encryption scheme
            has a separate public key for encrypting data, and a separate secret key for
            decrypting data. This way multiple parties can encrypt data using the same
            shared public key, but only the proper recipient of the data can decrypt it
            with the secret key.
			Microsoft SEAL中的加密方案是公钥加密方案。 对于不熟悉此术语的用户，公共密钥加密方案具有用于加密数据的单独的公共密钥和用于解密数据的单独的秘密密钥。
			这样，多方可以使用相同的共享公共密钥来加密数据，但是只有数据的正确接收者才能使用密钥对其进行解密。
            We are now ready to generate the secret and public keys. For this purpose
            we need an instance of the KeyGenerator class. Constructing a KeyGenerator
            automatically generates the public and secret key, which can immediately be
            read to local variables.
			现在，我们准备生成秘密和公共密钥。 为此，我们需要一个KeyGenerator类的实例。
			构造KeyGenerator会自动生成公用密钥和私有密钥，可以立即将其读取为局部变量。
            */
            KeyGenerator keygen = new KeyGenerator(context);
            PublicKey publicKey = keygen.PublicKey;
            SecretKey secretKey = keygen.SecretKey;

            /*
            To be able to encrypt we need to construct an instance of Encryptor. Note
            that the Encryptor only requires the public key, as expected.
			为了能够加密，我们需要构造一个Encryptor实例。 请注意，加密器只需要公用密钥即可。
            */
            Encryptor encryptor = new Encryptor(context, publicKey);

            /*
            Computations on the ciphertexts are performed with the Evaluator class. In
            a real use-case the Evaluator would not be constructed by the same party
            that holds the secret key.
			密文的计算是通过Evaluator类执行的。 在实际用例中，不会由拥有密钥的同一方来构造评估器。
            */
            Evaluator evaluator = new Evaluator(context);

            /*
            We will of course want to decrypt our results to verify that everything worked,
            so we need to also construct an instance of Decryptor. Note that the Decryptor
            requires the secret key.当然，我们将希望解密结果以验证一切正常，因此我们还需要构造一个Decryptor实例。 请注意，解密器需要私钥。
            */
            Decryptor decryptor = new Decryptor(context, secretKey);

            /*
            As an example, we evaluate the degree 4 polynomial 
            例如，我们评估4层多项式
                4x^4 + 8x^3 + 8x^2 + 8x + 4
            over an encrypted x = 6. The coefficients of the polynomial can be considered
            as plaintext inputs, as we will see below. The computation is done modulo the
            plain_modulus 1024.多项式的系数可以视为纯文本输入，如下所示。 计算以plain_modulus 1024为模。
            While this examples is simple and easy to understand, it does not have much
            practical value. In later examples we will demonstrate how to compute more
            efficiently on encrypted integers and real or complex numbers.
            Plaintexts in the BFV scheme are polynomials of degree less than the degree
            of the polynomial modulus, and coefficients integers modulo the plaintext
            modulus. 尽管此示例简单易懂，但并没有太大的实用价值。 在后面的示例中，我们将演示如何在加密的整数以及实数或复数上更有效地进行计算。
			BFV方案中的明文是次数小于多项式模数的次数的多项式，并且系数是对明文模量取模的整数。
			For readers with background in ring theory, the plaintext space is
            the polynomial quotient ring Z_T[X]/(X^N + 1), where N is PolyModulusDegree
            and T is PlainModulus.对于具有环论背景的读者来说，纯文本空间是多项式商环Z_T [X] /（X ^ N + 1），
			其中N是PolyModulusDegree，T是PlainModulus
            To get started, we create a plaintext containing the constant 6. For the
            plaintext element we use a constructor that takes the desired polynomial as
            a string with coefficients represented as hexadecimal numbers.
			首先，我们创建一个包含常量6的纯文本。对于纯文本元素，我们使用一个构造函数，
			该构造函数将所需的多项式作为字符串，并将系数表示为十六进制数。
            */
            Utilities.PrintLine();
            int x = 6;
            Plaintext xPlain = new Plaintext(x.ToString());
            Console.WriteLine($"Express x = {x} as a plaintext polynomial 0x{xPlain}.");

            /*
            We then encrypt the plaintext, producing a ciphertext.
            */
            Utilities.PrintLine();
            Ciphertext xEncrypted = new Ciphertext();
            Console.WriteLine("Encrypt xPlain to xEncrypted.");
            encryptor.Encrypt(xPlain, xEncrypted);

            /*
            In Microsoft SEAL, a valid ciphertext consists of two or more polynomials
            whose coefficients are integers modulo the product of the primes in the
            coeff_modulus. The number of polynomials in a ciphertext is called its `size'
            and is given by Ciphertext.Size. A freshly encrypted ciphertext always has
            size 2.在MicrosoftSeal中，一个有效的密文由两个或多个多项式组成，这些多项式的系数是在Coeff_模数中模数乘积的整数。
			密文中的多项式的数目称为其“大小”，
			并且由ciphertext.size.给出，新加密的密文总是具有大小2
            */
            Console.WriteLine($"    + size of freshly encrypted x: {xEncrypted.Size}");

            /*
            There is plenty of noise budget left in this freshly encrypted ciphertext.
			在这个新加密的密文中留下了大量的噪声预算。
            */
            Console.WriteLine("    + noise budget in freshly encrypted x: {0} bits",
                decryptor.InvariantNoiseBudget(xEncrypted));

            /*
            We decrypt the ciphertext and print the resulting plaintext in order to
            demonstrate correctness of the encryption.我们解密密文并打印得到的明文，以证明加密的正确性。
            */
            Plaintext xDecrypted = new Plaintext();
            Console.Write("    + decryption of encrypted_x: ");
            decryptor.Decrypt(xEncrypted, xDecrypted);
            Console.WriteLine($"0x{xDecrypted} ...... Correct.");

            /*
            When using Microsoft SEAL, it is typically advantageous to compute in a way
            that minimizes the longest chain of sequential multiplications. In other
            words, encrypted computations are best evaluated in a way that minimizes
            the multiplicative depth of the computation, because the total noise budget
            consumption is proportional to the multiplicative depth. For example, for
            our example computation it is advantageous to factorize the polynomial as当使用Microsoft密封件时，通常有利的是以最小化顺序乘法的最长链的方式来计算。换句话说，以
			最小化计算的乘法深度的方式来最佳地评估加密的计算，因为总噪声预算消耗与乘法深度成比例。
			例如，对于我们的示例计算，将多项式分解为有利的是有利的。
                4x^4 + 8x^3 + 8x^2 + 8x + 4 = 4(x + 1)^2 * (x^2 + 1)
            to obtain a simple depth 2 representation. Thus, we compute (x + 1)^2 and
            (x^2 + 1) separately, before multiplying them, and multiplying by 4.
            First, we compute x^2 and add a plaintext "1". We can clearly see from the
            print-out that multiplication has consumed a lot of noise budget. The user
            can vary the plain_modulus parameter to see its effect on the rate of noise
            budget consumption.若要获得简单的深度2表示，请执行以下操作。因此，我们分别计算(X1)^2和(x^2.1)，然后再将它们乘以4。
           首先，我们计算x^2并添加一个明文“1”。我们可以从打印中清楚地看到，乘法消耗了大量的噪声预算。用户可以改变plain_modulus参数，以查看其对噪声预算消耗率的影响。
		   */
            Utilities.PrintLine();
            Console.WriteLine("Compute xSqPlusOne (x^2+1).");
            Ciphertext xSqPlusOne = new Ciphertext();
            evaluator.Square(xEncrypted, xSqPlusOne);
            Plaintext plainOne = new Plaintext("1");
            evaluator.AddPlainInplace(xSqPlusOne, plainOne);

            /*
            Encrypted multiplication results in the output ciphertext growing in size.
            More precisely, if the input ciphertexts have size M and N, then the output
            ciphertext after homomorphic multiplication will have size M+N-1. In this
            case we perform a squaring, and observe both size growth and noise budget
            consumption.加密乘法会导致输出密文的大小增大。更准确地说，如果输入密文的大小为m和n，则同态乘法后的输出密文的大小为mn-1。
			在这种情况下，我们执行平方，并观察大小增长和噪声预算消耗。
            */
            Console.WriteLine($"    + size of xSqPlusOne: {xSqPlusOne.Size}");
            Console.WriteLine("    + noise budget in xSqPlusOne: {0} bits",
                decryptor.InvariantNoiseBudget(xSqPlusOne));

            /*
            Even though the size has grown, decryption works as usual as long as noise
            budget has not reached 0.即使大小增加了，只要噪声预算没有达到0，解密就会像往常一样工作。
            */
            Plaintext decryptedResult = new Plaintext();
            Console.Write("    + decryption of xSqPlusOne: ");
            decryptor.Decrypt(xSqPlusOne, decryptedResult);
            Console.WriteLine($"0x{decryptedResult} ...... Correct.");

            /*
            Next, we compute (x + 1)^2.
            */
            Utilities.PrintLine();
            Console.WriteLine("Compute xPlusOneSq ((x+1)^2).");
            Ciphertext xPlusOneSq = new Ciphertext();
            evaluator.AddPlain(xEncrypted, plainOne, xPlusOneSq);
            evaluator.SquareInplace(xPlusOneSq);
            Console.WriteLine($"    + size of xPlusOneSq: {xPlusOneSq.Size}");
            Console.WriteLine("    + noise budget in xPlusOneSq: {0} bits",
                decryptor.InvariantNoiseBudget(xPlusOneSq));
            Console.Write("    + decryption of xPlusOneSq: ");
            decryptor.Decrypt(xPlusOneSq, decryptedResult);
            Console.WriteLine($"0x{decryptedResult} ...... Correct.");

            /*
            Finally, we multiply (x^2 + 1) * (x + 1)^2 * 4.
            */
            Utilities.PrintLine();
            Console.WriteLine("Compute encryptedResult (4(x^2+1)(x+1)^2).");
            Ciphertext encryptedResult = new Ciphertext();
            Plaintext plainFour = new Plaintext("4");
            evaluator.MultiplyPlainInplace(xSqPlusOne, plainFour);
            evaluator.Multiply(xSqPlusOne, xPlusOneSq, encryptedResult);
            Console.WriteLine($"    + size of encrypted_result: {encryptedResult.Size}");
            Console.WriteLine("    + noise budget in encrypted_result: {0} bits",
                decryptor.InvariantNoiseBudget(encryptedResult));
            Console.WriteLine("NOTE: Decryption can be incorrect if noise budget is zero.");

            Console.WriteLine();
            Console.WriteLine("~~~~~~ A better way to calculate 4(x^2+1)(x+1)^2. ~~~~~~");

            /*
            Noise budget has reached 0, which means that decryption cannot be expected
            to give the correct result. This is because both ciphertexts xSqPlusOne and
            xPlusOneSq consist of 3 polynomials due to the previous squaring operations,
            and homomorphic operations on large ciphertexts consume much more noise budget
            than computations on small ciphertexts. Computing on smaller ciphertexts is
            also computationally significantly cheaper.噪声预算已经达到0，这意味着不能预期解密给出正确的结果。这是因为加密文本xsqlusone和xplusesq
			都由3个多项式组成，
			这是因为先前的平方运算，而在大的密文上的同态操作消耗比在小密文上的计算更多的噪声预算。
			在较小的密文上的计算也在计算上显著地便宜,
            `Relinearization' is an operation that reduces the size of a ciphertext after
            multiplication back to the initial size, 2. Thus, relinearizing one or both
            input ciphertexts before the next multiplication can have a huge positive
            impact on both noise growth and performance, even though relinearization has
            a significant computational cost itself. It is only possible to relinearize
            size 3 ciphertexts down to size 2, so often the user would want to relinearize
            after each multiplication to keep the ciphertext sizes at 2.“再线性化”是一种操作,其在乘法回到初始
			大小2之后减小密文的大小。因此，即使重新线性化具有显著的计算成本本身，在下一个乘法之前对一个或两个输入密文重新线性化
			可以对噪声增长和性能产生巨大的积极影响。仅有可能将大小3的密文重新线性化到大小2，
			因此通常用户希望在每次相乘之后重新线性化，以保持密文大小为2。
            Relinearization requires special `relinearization keys', which can be thought
            of as a kind of public key. Relinearization keys can easily be created with
            the KeyGenerator.
            Relinearization is used similarly in both the BFV and the CKKS schemes, but
            in this example we continue using BFV. We repeat our computation from before,
            but this time relinearize after every multiplication.在BfV和ckks方案中都使用了类似的再定位，但是在本例中我们继续使用BfV。
			我们重复以前的计算，但这一次在每一次乘法之后再进行再分类
            We use KeyGenerator.RelinKeys() to create relinearization keys.
            */
            Utilities.PrintLine();
            Console.WriteLine("Generate relinearization keys.");
            RelinKeys relinKeys = keygen.RelinKeys();

            /*
            We now repeat the computation relinearizing after each multiplication.我们现在在每次乘法后重复计算。
            */
            Utilities.PrintLine();
            Console.WriteLine("Compute and relinearize xSquared (x^2),");
            Console.WriteLine(new string(' ', 13) + "then compute xSqPlusOne (x^2+1)");
            Ciphertext xSquared = new Ciphertext();
            evaluator.Square(xEncrypted, xSquared);
            Console.WriteLine($"    + size of xSquared: {xSquared.Size}");
            evaluator.RelinearizeInplace(xSquared, relinKeys);
            Console.WriteLine("    + size of xSquared (after relinearization): {0}",
                xSquared.Size);
            evaluator.AddPlain(xSquared, plainOne, xSqPlusOne);
            Console.WriteLine("    + noise budget in xSqPlusOne: {0} bits",
                decryptor.InvariantNoiseBudget(xSqPlusOne));
            Console.Write("    + decryption of xSqPlusOne: ");
            decryptor.Decrypt(xSqPlusOne, decryptedResult);
            Console.WriteLine($"0x{decryptedResult} ...... Correct.");

            Utilities.PrintLine();
            Ciphertext xPlusOne = new Ciphertext();
            Console.WriteLine("Compute xPlusOne (x+1),");
            Console.WriteLine(new string(' ', 13) +
                "then compute and relinearize xPlusOneSq ((x+1)^2).");
            evaluator.AddPlain(xEncrypted, plainOne, xPlusOne);
            evaluator.Square(xPlusOne, xPlusOneSq);
            Console.WriteLine($"    + size of xPlusOneSq: {xPlusOneSq.Size}");
            evaluator.RelinearizeInplace(xPlusOneSq, relinKeys);
            Console.WriteLine("    + noise budget in xPlusOneSq: {0} bits",
                decryptor.InvariantNoiseBudget(xPlusOneSq));
            Console.Write("    + decryption of xPlusOneSq: ");
            decryptor.Decrypt(xPlusOneSq, decryptedResult);
            Console.WriteLine($"0x{decryptedResult} ...... Correct.");

            Utilities.PrintLine();
            Console.WriteLine("Compute and relinearize encryptedResult (4(x^2+1)(x+1)^2).");
            evaluator.MultiplyPlainInplace(xSqPlusOne, plainFour);
            evaluator.Multiply(xSqPlusOne, xPlusOneSq, encryptedResult);
            Console.WriteLine($"    + size of encryptedResult: {encryptedResult.Size}");
            evaluator.RelinearizeInplace(encryptedResult, relinKeys);
            Console.WriteLine("    + size of encryptedResult (after relinearization): {0}",
                encryptedResult.Size);
            Console.WriteLine("    + noise budget in encryptedResult: {0} bits",
                decryptor.InvariantNoiseBudget(encryptedResult));

            Console.WriteLine();
            Console.WriteLine("NOTE: Notice the increase in remaining noise budget.");

            /*
            Relinearization clearly improved our noise consumption. We have still plenty
            of noise budget left, so we can expect the correct answer when decrypting.再碱化明显改善了我们的噪音消耗。我们仍然有足够的噪音预算，所以我们可以期待正确的答案时，解密。
            */
            Utilities.PrintLine();
            Console.WriteLine("Decrypt encrypted_result (4(x^2+1)(x+1)^2).");
            decryptor.Decrypt(encryptedResult, decryptedResult);
            Console.WriteLine("    + decryption of 4(x^2+1)(x+1)^2 = 0x{0} ...... Correct.",
                decryptedResult);

            /*
            For x=6, 4(x^2+1)(x+1)^2 = 7252. Since the plaintext modulus is set to 1024,
            this result is computed in integers modulo 1024. Therefore the expected output
            should be 7252 % 1024 == 84, or 0x54 in hexadecimal.X=6,4(x,21)(x1),2=7252。由于明文模数被设置为1024，所以该结果以整数模数1024计算。因此，预期输出应为7252%1024==84，或十六进制的0x54。
            */
        }
    }
}
