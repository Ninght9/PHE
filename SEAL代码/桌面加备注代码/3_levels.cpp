// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

using System;
using Microsoft.Research.SEAL;

namespace SEALNetExamples
{
    partial class Examples
    {
        private static void ExampleLevels()
        {
            Utilities.PrintExampleBanner("Example: Levels");

            /*
            In this examples we describe the concept of `levels' in BFV and CKKS and the
            related objects that represent them in Microsoft SEAL.在此示例中，我们描述了BFV和CKS中“级别”的
			概念以及在Microsoft密封件中表示它们的相关对象。
            In Microsoft SEAL a set of encryption parameters (excluding the random number
            generator) is identified uniquely by a 256-bit hash of the parameters. This
            hash is called the `ParmsId' and can be easily accessed and printed at any
            time.n Microsoft密封一组加密参数(不包括随机数生成器)是由参数的256位哈希唯一标识的。
			这个散列名为“parmsid”，可以随时方便地访问和打印。 
			The hash will change as soon as any of the parameters is changed.
            When a SEALContext is created from a given EncryptionParameters instance,
            Microsoft SEAL automatically creates a so-called `modulus switching chain',
            which is a chain of other encryption parameters derived from the original set.
            The parameters in the modulus switching chain are the same as the original
            parameters with the exception that size of the coefficient modulus is
            decreasing going down the chain. More precisely, each parameter set in the
            chain attempts to remove the last coefficient modulus prime from the
            previous set; this continues until the parameter set is no longer valid
            (e.g., PlainModulus is larger than the remaining CoeffModulus). It is easy
            to walk through the chain and access all the parameter sets. Additionally,
            each parameter set in the chain has a `chain index' that indicates its
            position in the chain so that the last set has index 0. We say that a set
            of encryption parameters, or an object carrying those encryption parameters,
            is at a higher level in the chain than another set of parameters if its the
            chain index is bigger, i.e., it is earlier in the chain.
            Each set of parameters in the chain involves unique pre-computations performed
            when the SEALContext is created, and stored in a SEALContext.ContextData
            object. The chain is basically a linked list of SEALContext.ContextData
            objects, and can easily be accessed through the SEALContext at any time. Each
            node can be identified by the ParmsId of its specific encryption parameters
            (PolyModulusDegree remains the same but CoeffModulus varies).只要更改了任何参数，哈希值就会改变。
			当从给定的EncryptionParameters实例创建SEALContext时，Microsoft SEAL自动创建一个所谓的“模交换链”，
			该链是从加密参数派生的其他加密参数链。原始集。模数转换链中的参数与原始参数相同，不同之处在于，系数模数的大小沿
			链向下逐渐减小。更准确地说，链中的每个参数集都尝试从上一个参数集中删除最后一个系数模质数。这将继续进行直到参数集
			不再有效为止（例如，PlainModulus大于其余的CoeffModulus）。轻松浏览链条并访问所有参数集。此外，链中的每个参数集都有一个“
			链索引”，用于指示其在链中的位置，因此最后一个参数集的索引为0。我们说一组加密参数或一个带有这些加密参数的对象位于一个位置。
			如果链索引较大，即链中较早，则链中的级别比另一组参数高。链中的每组参数都涉及创建SEALContext时执行的唯一预计算，并将其存储
			在SEALContext.ContextData中。
 宾语。该链基本上是SEALContext.ContextData对象的链接列表，并且可以随时通过SEALContext轻松访问。每个节点都可以通过其特定加密参
数的ParmsId进行标识（PolyModulusDegree保持不变，但CoeffModulus有所不同）。
            */
            EncryptionParameters parms = new EncryptionParameters(SchemeType.BFV);
            ulong polyModulusDegree = 8192;
            parms.PolyModulusDegree = polyModulusDegree;

            /*
            In this example we use a custom CoeffModulus, consisting of 5 primes of
            sizes 50, 30, 30, 50, and 50 bits. Note that this is still OK according to
            the explanation in `1_BFV_Basics.cs'. Indeed,在本例中，我们使用自定义系数，由大小为50、30、30、50和50位
			的5个素数组成。注意，根据‘1_BfV_basics.cs’中的解释，这仍然是可以的。的确
                CoeffModulus.MaxBitCount(polyModulusDegree)
            returns 218 (greater than 50+30+30+50+50=210).
            Due to the modulus switching chain, the order of the 5 primes is significant.
            The last prime has a special meaning and we call it the `special prime'. Thus,
            the first parameter set in the modulus switching chain is the only one that
            involves the special prime. All key objects, such as SecretKey, are created
            at this highest level. All data objects, such as Ciphertext, can be only at
            lower levels. The special modulus should be as large as the largest of the
            other primes in the CoeffModulus, although this is not a strict requirement.
			由于模量交换链，5个素数的顺序是显著的。最后一个素数有一个特殊的意义，我们称之为“特殊素数”。
			因此，在模数交换链中设置的第一参数是唯一涉及特殊素数的参数。所有关键对象（如secretkey）都在此最高级别创建。
			所有数据对象（例如密文）只能在较低的级别。尽管这不是严格的要求，但特殊模量应该与系数中的其它素数的最大值一样大
                     special prime +---------+
                                             |
                                             v
            CoeffModulus: { 50, 30, 30, 50, 50 }  +---+  Level 4 (all keys; `key level')
                                                      |
                                                      |
                CoeffModulus: { 50, 30, 30, 50 }  +---+  Level 3 (highest `data level')
                                                      |
                                                      |
                    CoeffModulus: { 50, 30, 30 }  +---+  Level 2
                                                      |
                                                      |
                        CoeffModulus: { 50, 30 }  +---+  Level 1
                                                      |
                                                      |
                            CoeffModulus: { 50 }  +---+  Level 0 (lowest level)
            */
            parms.CoeffModulus = CoeffModulus.Create(
                polyModulusDegree, new int[] { 50, 30, 30, 50, 50 });

            /*
            In this example the PlainModulus does not play much of a role; we choose
            some reasonable value.在这个例子中，明文模并不起很大的作用；我们选择了一些合理的值。
            */
            parms.PlainModulus = PlainModulus.Batching(polyModulusDegree, 20);

            SEALContext context = new SEALContext(parms);
            Utilities.PrintParameters(context);

            /*
            There are convenience method for accessing the SEALContext.ContextData for
            some of the most important levels:对于一些最重要的级别，有一些访问sealcontext.contextdata
			的方便方法：
                SEALContext.KeyContextData: access to key level ContextData
                SEALContext.FirstContextData: access to highest data level ContextData
                SEALContext.LastContextData: access to lowest level ContextData
            We iterate over the chain and print the ParmsId for each set of parameters.我们遍历链并为每组参数打印parmsd。
            */
            Console.WriteLine();
            Utilities.PrintLine();
            Console.WriteLine("Print the modulus switching chain.");

            /*
            First print the key level parameter information.首先打印键级参数信息。
            */
            SEALContext.ContextData contextData = context.KeyContextData;
            Console.WriteLine("----> Level (chain index): {0} ...... KeyContextData",
                contextData.ChainIndex);
            Console.WriteLine($"      ParmsId: {contextData.ParmsId}");
            Console.Write("      CoeffModulus primes: ");
            foreach (SmallModulus prime in contextData.Parms.CoeffModulus)
            {
                Console.Write($"{Utilities.ULongToString(prime.Value)} ");
            }
            Console.WriteLine();
            Console.WriteLine("\\");
            Console.Write(" \\--> ");

            /*
            Next iterate over the remaining (data) levels.
            */
            contextData = context.FirstContextData;
            while (null != contextData)
            {
                Console.Write($"Level (chain index): {contextData.ChainIndex}");
                if (contextData.ParmsId.Equals(context.FirstParmsId))
                {
                    Console.WriteLine(" ...... FirstContextData");
                }
                else if (contextData.ParmsId.Equals(context.LastParmsId))
                {
                    Console.WriteLine(" ...... LastContextData");
                }
                else
                {
                    Console.WriteLine();
                }
                Console.WriteLine($"      ParmsId: {contextData.ParmsId}");
                Console.Write("      CoeffModulus primes: ");
                foreach (SmallModulus prime in contextData.Parms.CoeffModulus)
                {
                    Console.Write($"{Utilities.ULongToString(prime.Value)} ");
                }
                Console.WriteLine();
                Console.WriteLine("\\");
                Console.Write(" \\--> ");

                /*
                Step forward in the chain.
                */
                contextData = contextData.NextContextData;
            }
            Console.WriteLine("End of chain reached");
            Console.WriteLine();

            /*
            We create some keys and check that indeed they appear at the highest level.我们创建一些关键点并检查它们是否出现在最高级别上。
            */
            KeyGenerator keygen = new KeyGenerator(context);
            PublicKey publicKey = keygen.PublicKey;
            SecretKey secretKey = keygen.SecretKey;
            RelinKeys relinKeys = keygen.RelinKeys();
            GaloisKeys galoisKeys = keygen.GaloisKeys();
            Utilities.PrintLine();
            Console.WriteLine("Print the parameter IDs of generated elements.");
            Console.WriteLine($"    + publicKey:  {publicKey.ParmsId}");
            Console.WriteLine($"    + secretKey:  {secretKey.ParmsId}");
            Console.WriteLine($"    + relinKeys:  {relinKeys.ParmsId}");
            Console.WriteLine($"    + galoisKeys: {galoisKeys.ParmsId}");

            Encryptor encryptor = new Encryptor(context, publicKey);
            Evaluator evaluator = new Evaluator(context);
            Decryptor decryptor = new Decryptor(context, secretKey);

            /*
            In the BFV scheme plaintexts do not carry a ParmsId, but ciphertexts do. Note
            how the freshly encrypted ciphertext is at the highest data level.
			在BFV方案中，明文不为携带parmsid，但密文do。注意新加密的密文是如何处于最高数据级别的。
            */
            Plaintext plain = new Plaintext("1x^3 + 2x^2 + 3x^1 + 4");
            Ciphertext encrypted = new Ciphertext();
            encryptor.Encrypt(plain, encrypted);
            Console.WriteLine($"    + plain:      {plain.ParmsId} (not set in BFV)");
            Console.WriteLine($"    + encrypted:  {encrypted.ParmsId}");
            Console.WriteLine();

            /*
            `Modulus switching' is a technique of changing the ciphertext parameters down
            in the chain. The function Evaluator.ModSwitchToNext always switches to the
            next level down the chain, whereas Evaluator.ModSwitchTo switches to a parameter
            set down the chain corresponding to a given ParmsId. However, it is impossible
            to switch up in the chain.“模数切换”是一种改变链中密文参数的技术。Moditionator.modSwittonext函数总是切换到链下的下一个级别，
			而计算器.modSwitto则切换到与给定的parmsid对应的链下设置的参数。然而，在链中切换是不可能的。
            */
            Utilities.PrintLine();
            Console.WriteLine("Perform modulus switching on encrypted and print.");
            contextData = context.FirstContextData;
            Console.Write("----> ");
            while (null != contextData.NextContextData)
            {
                Console.WriteLine($"Level (chain index): {contextData.ChainIndex}");
                Console.WriteLine($"      ParmsId of encrypted: {contextData.ParmsId}");
                Console.WriteLine("      Noise budget at this level: {0} bits",
                    decryptor.InvariantNoiseBudget(encrypted));
                Console.WriteLine("\\");
                Console.Write(" \\--> ");
                evaluator.ModSwitchToNextInplace(encrypted);
                contextData = contextData.NextContextData;
            }
            Console.WriteLine($"Level (chain index): {contextData.ChainIndex}");
            Console.WriteLine($"      ParmsId of encrypted: {contextData.ParmsId}");
            Console.WriteLine("      Noise budget at this level: {0} bits",
                decryptor.InvariantNoiseBudget(encrypted));
            Console.WriteLine("\\");
            Console.Write(" \\--> ");
            Console.WriteLine("End of chain reached");
            Console.WriteLine();

            /*
            At this point it is hard to see any benefit in doing this: we lost a huge
            amount of noise budget (i.e., computational power) at each switch and seemed
            to get nothing in return. Decryption still works.在这一点上，很难看到任何好处，这样做：我们失去了巨大的噪音预算(即计算能力)在每个开关，
			似乎没有得到任何回报。解密仍然有效。
            */
            Utilities.PrintLine();
            Console.WriteLine("Decrypt still works after modulus switching.");
            decryptor.Decrypt(encrypted, plain);
            Console.WriteLine($"    + Decryption of encrypted: {plain} ...... Correct.");
            Console.WriteLine();

            /*
            However, there is a hidden benefit: the size of the ciphertext depends
            linearly on the number of primes in the coefficient modulus. Thus, if there
            is no need or intention to perform any further computations on a given
            ciphertext, we might as well switch it down to the smallest (last) set of
            parameters in the chain before sending it back to the secret key holder for
            decryption.然而，有一个隐藏的好处：密文的大小与系数模数中素数的大小成线性关系。因此，
			如果不需要或不打算对给定的密文执行任何进一步的计算，
			我们也可以将其切换到链中最小(最后一组)的参数集，然后再将其发送回秘密密钥持有者进行解密。
            Also the lost noise budget is actually not an issue at all, if we do things
            right, as we will see below.此外，损失的噪音预算实际上根本不是一个问题，如果我们做正确的事情，我们将看到下面。
            First we recreate the original ciphertext and perform some computations.首先，我们重新创建原始密文并执行一些计算。
            */
            Console.WriteLine("Computation is more efficient with modulus switching.");
            Utilities.PrintLine();
            Console.WriteLine("Compute the eight power.");
            encryptor.Encrypt(plain, encrypted);
            Console.WriteLine("    + Noise budget fresh:                  {0} bits",
                decryptor.InvariantNoiseBudget(encrypted));
            evaluator.SquareInplace(encrypted);
            evaluator.RelinearizeInplace(encrypted, relinKeys);
            Console.WriteLine("    + Noise budget of the 2nd power:        {0} bits",
                decryptor.InvariantNoiseBudget(encrypted));
            evaluator.SquareInplace(encrypted);
            evaluator.RelinearizeInplace(encrypted, relinKeys);
            Console.WriteLine("    + Noise budget of the 4th power:        {0} bits",
                decryptor.InvariantNoiseBudget(encrypted));
            /*
            Surprisingly, in this case modulus switching has no effect at all on the
            noise budget.令人惊讶的是，在这种情况下，模量切换对噪声预算没有影响。
            */
            evaluator.ModSwitchToNextInplace(encrypted);
            Console.WriteLine("    + Noise budget after modulus switching: {0} bits",
                decryptor.InvariantNoiseBudget(encrypted));


            /*
            This means that there is no harm at all in dropping some of the coefficient
            modulus after doing enough computations. In some cases one might want to
            switch to a lower level slightly earlier, actually sacrificing some of the
            noise budget in the process, to gain computational performance from having
            smaller parameters. We see from the print-out that the next modulus switch
            should be done ideally when the noise budget is down to around 25 bits.这意味着，在做了足够的计算后，
			降低一些系数模数是没有害处的。在某些情况下，人们可能希望稍微早一点切换到较低的级别，
			实际上在这个过程中牺牲了一些噪声预算，以便从较小的参数中获得计算性能。
			我们从打印结果中看到，下一个模数开关应该在噪音预算降到25位左右时进行。
            */
            evaluator.SquareInplace(encrypted);
            evaluator.RelinearizeInplace(encrypted, relinKeys);
            Console.WriteLine("    + Noise budget of the 8th power:        {0} bits",
                decryptor.InvariantNoiseBudget(encrypted));
            evaluator.ModSwitchToNextInplace(encrypted);
            Console.WriteLine("    + Noise budget after modulus switching: {0} bits",
                decryptor.InvariantNoiseBudget(encrypted));

            /*
            At this point the ciphertext still decrypts correctly, has very small size,
            and the computation was as efficient as possible. Note that the decryptor
            can be used to decrypt a ciphertext at any level in the modulus switching
            chain.此时密文仍然正确解密，体积很小，计算效率尽可能高。注意，解密器可用于
			解密模数交换链中任意级别的密文。
            */
            decryptor.Decrypt(encrypted, plain);
            Console.WriteLine("    + Decryption of the 8th power (hexadecimal) ...... Correct.");
            Console.WriteLine($"    {plain}");
            Console.WriteLine();

            /*
            In BFV modulus switching is not necessary and in some cases the user might
            not want to create the modulus switching chain, except for the highest two
            levels. This can be done by passing a bool `false' to SEALContext constructor.在BfV模数交换是不必要的，在某些情况下，
			用户可能不想创建模数交换链，除了最高的两个级别。这可以通过将bool‘false’传递给密封上下文构造函数来实现。
            */
            context = new SEALContext(parms, expandModChain: false);

            /*
            We can check that indeed the modulus switching chain has been created only
            for the highest two levels (key level and highest data level). The following
            loop should execute only once.我们可以检查模数切换链确实只为最高的两个级别(键级和最高数据级别)创建。下面的循环应该只执行一次。
            */
            Console.WriteLine("Optionally disable modulus switching chain expansion.");
            Utilities.PrintLine();
            Console.WriteLine("Print the modulus switching chain.");
            Console.Write("----> ");
            for (contextData = context.KeyContextData; null != contextData;
                contextData = contextData.NextContextData)
            {
                Console.WriteLine($"Level (chain index): {contextData.ChainIndex}");
                Console.WriteLine($"      ParmsId of encrypted: {contextData.ParmsId}");
                Console.Write("      CoeffModulus primes: ");
                foreach (SmallModulus prime in contextData.Parms.CoeffModulus)
                {
                    Console.Write($"{Utilities.ULongToString(prime.Value)} ");
                }
                Console.WriteLine();
                Console.WriteLine("\\");
                Console.Write(" \\--> ");
            }
            Console.WriteLine("End of chain reached");
            Console.WriteLine();

            /*
            It is very important to understand how this example works since in the CKKS
            scheme modulus switching has a much more fundamental purpose and the next
            examples will be difficult to understand unless these basic properties are
            totally clear.了解这个例子是非常重要的，因为在CKKS方案中，模数切换有一个更基本的目的，
			除非这些基本性质完全清楚，否则下一个例子将很难理解。
            */
        }
    }
}