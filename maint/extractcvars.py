#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from ruamel.yaml import YAML
import os
from os import walk
from io import StringIO

acceptedEnvs = [
    ]

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


def loadCvarsFromFiles(filenames):
    stream = "cvars:\n"
    for file in filenames:
        f = open(file, 'r')
        lines = f.readlines()
        f.close()

        ready = False
        for line in lines:
            if line.find('END_NCCL_CVAR_INFO_BLOCK') != -1:
                ready = False
            if ready == True:
                stream += line
            if line.find('BEGIN_NCCL_CVAR_INFO_BLOCK') != -1:
                ready = True

    return YAML().load(stream)


@static_vars(counter = 0)
def indent(file, str_):
    str = str_.strip()
    if (str[0] == '}'):
        c = indent.counter - 1
    else:
        c = indent.counter
    spaces = ""
    for i in range(c):
        spaces += "  "
    file.write("%s%s\n" % (spaces, str))
    indent.counter += str.count('{') - str.count('}')


class basetype:
    def __init__(self, cvar):
        self.name = cvar['name']
        self.default = cvar['default']
        self.description = cvar['description']
        self.type = cvar['type']
        if 'envstr' in cvar:
            self.envstr = cvar['envstr']
        else:
            self.envstr = self.name
        if 'choices' in cvar:
            self.choices = cvar['choices']
        else:
            self.choices = ""
        if 'prefixes' in cvar:
            self.prefixes = cvar['prefixes']
        else:
            self.prefixes = ""

    @staticmethod
    def utilfns(file):
        pass

    def externDecl(self, file):
        indent(file, "extern %s %s;" % (self.type, self.name))
        indent(file, "extern %s %s_DEFAULT;" % (self.type, self.name))
        file.write("\n")

    def storageDecl(self, file):
        indent(file, "%s %s;" % (self.type, self.name))
        indent(file, "%s %s_DEFAULT;" % (self.type, self.name))

    def desc(self, file):
        file.write("\n")
        if self.name == self.envstr:
            file.write("%s\n" % self.name)
        else:
            file.write("%s (internal variable within NCCL: %s)\n" %
                (self.envstr, self.name))
        file.write("Description:\n")
        d = self.description.split("\n")
        for line in d:
            file.write("    %s\n" % line)
        file.write("Type: %s\n" % self.type)
        file.write("Default: %s\n" % self.default)

    def unknownValUnitTest(self, file):
        indent(file, "TEST_F(CvarTest, %s_warn_unknown_val) {" % (self.name))
        indent(file, "setenv(\"%s\", \"dummy\", 1);" % (self.envstr))
        indent(file, "testWarn(\"%s\", \"Unknown value\");" % (self.envstr))
        indent(file, "}")
        file.write("\n")

    def dupValUnitTest(self, file):
        indent(file, "TEST_F(CvarTest, %s_warn_dup_val) {" % (self.name))
        indent(file, "setenv(\"%s\", \"dummy,dummy\", 1);" % (self.envstr))
        indent(file, "testWarn(\"%s\", \"Duplicate token\");" % (self.envstr))
        indent(file, "}")
        file.write("\n")

class bool(basetype):
    @staticmethod
    def utilfns(file):
        pass

    def unitTest(self, file):
        for i, val in enumerate(["y", "yes", "true", "1"]):
            indent(file, "TEST_F(CvarTest, %s_value_y%s) {" % (self.name, i))
            indent(file, "setenv(\"%s\", \"%s\", 1);" % (self.envstr, val))
            indent(file, "ncclCvarInit();")
            indent(file, "EXPECT_TRUE(%s);" % (self.name))
            indent(file, "}")
            file.write("\n")

        for i, val in enumerate(["n", "no", "false", "0"]):
            indent(file, "TEST_F(CvarTest, %s_value_n%s) {" % (self.name, i))
            indent(file, "setenv(\"%s\", \"%s\", 1);" % (self.envstr, val))
            indent(file, "ncclCvarInit();")
            indent(file, "EXPECT_FALSE(%s);" % (self.name))
            indent(file, "}")
            file.write("\n")

        if self.default:
            indent(file, "TEST_F(CvarTest, %s_default_value) {" % (self.name))
            indent(file, "testDefaultValue(\"%s\");" % (self.envstr))
            func = "EXPECT_TRUE" if self.default == True else "EXPECT_FALSE"
            indent(file, "%s(%s);" % (func, self.name))
            indent(file, "}")
            file.write("\n")

        self.unknownValUnitTest(file)

    def readenv(self, file):
        indent(file, "%s = env2bool(\"%s\", \"%s\");" %
            (self.name, self.envstr, self.default))
        indent(file, "%s_DEFAULT = env2bool(\"NCCL_ENV_DO_NOT_SET\", \"%s\");" %
            (self.name, self.default))
        file.write("\n")


class numeric(basetype):
    @staticmethod
    def utilfns(file):
        pass

    def unitTest(self, file):
        for i, val in enumerate(["0", "9999", \
            "std::numeric_limits<%s>::max()" % self.type, \
            "std::numeric_limits<%s>::min()" % self.type]):
            indent(file, "TEST_F(CvarTest, %s_value_%s) {" % (self.name, i))
            indent(file, "testNumValue<%s>(\"%s\", %s);" % (self.type, self.envstr, val))
            indent(file, "EXPECT_EQ(%s, %s);" % (self.name, val))
            indent(file, "}")
            file.write("\n")

        if self.default:
            indent(file, "TEST_F(CvarTest, %s_default_value) {" % (self.name))
            indent(file, "testDefaultValue(\"%s\");" % (self.envstr))
            if self.default == "MAX":
                indent(file, "EXPECT_EQ(%s, std::numeric_limits<%s>::max());" %
                    (self.name, self.type))
            elif self.default == "MIN":
                indent(file, "EXPECT_EQ(%s, std::numeric_limits<%s>::min());" %
                    (self.name, self.type))
            else:
                indent(file, "EXPECT_EQ(%s, %s);" % (self.name, self.default))
            indent(file, "}")
            file.write("\n")

    def readenv(self, file):
        indent(file, "%s = env2num<%s>(\"%s\", \"%s\");" %
            (self.name, self.type, self.envstr, self.default))
        indent(file, "%s_DEFAULT = env2num<%s>(\"NCCL_ENV_DO_NOT_SET\", \"%s\");" %
            (self.name, self.type, self.default))
        file.write("\n")


class string(basetype):
    @staticmethod
    def utilfns(file):
        pass

    def externDecl(self, file):
        indent(file, "extern std::string %s;" % self.name)
        indent(file, "extern std::string %s_DEFAULT;" % self.name)
        file.write("\n")

    def storageDecl(self, file):
        indent(file, "std::string %s;" % self.name)
        indent(file, "std::string %s_DEFAULT;" % self.name)

    def unitTest(self, file):
        for i, val in enumerate(["val1", "  val2_with_space   "]):
            indent(file, "TEST_F(CvarTest, %s_value_%s) {" % (self.name, i))
            indent(file, "setenv(\"%s\", \"%s\", 1);" % (self.envstr, val))
            indent(file, "ncclCvarInit();")
            indent(file, "EXPECT_EQ(%s, \"%s\");" % (self.name, val.strip()))
            indent(file, "}")
            file.write("\n")

        if self.default:
            indent(file, "TEST_F(CvarTest, %s_default_value) {" % (self.name))
            indent(file, "testDefaultValue(\"%s\");" % (self.envstr))
            indent(file, "EXPECT_EQ(%s, \"%s\");" % (self.name, self.default))
            indent(file, "}")
            file.write("\n")

    def readenv(self, file):
        default = self.default if self.default else ""
        indent(file, "%s = env2str(\"%s\", \"%s\");" %
            (self.name, self.envstr, default))
        indent(file, "%s_DEFAULT = env2str(\"NCCL_ENV_DO_NOT_SET\", \"%s\");" %
            (self.name, default))
        file.write("\n")


class stringlist(basetype):
    @staticmethod
    def utilfns(file):
        pass

    def externDecl(self, file):
        indent(file, "extern std::vector<std::string> %s;" % self.name)
        indent(file, "extern std::vector<std::string> %s_DEFAULT;" % self.name)
        file.write("\n")

    def storageDecl(self, file):
        indent(file, "std::vector<std::string> %s;" % self.name)
        indent(file, "std::vector<std::string> %s_DEFAULT;" % self.name)

    def unitTest(self, file):
        for i, val in enumerate(["val1,val2,val3", "val1:1,val2:2,val3:3", "val", "val1, val_w_space  "]):
            indent(file, "TEST_F(CvarTest, %s_valuelist_%s) {" % (self.name, i))
            indent(file, "setenv(\"%s\", \"%s\", 1);" % (self.envstr, val))
            trimmedVals = [v.strip() for v in val.split(",")]
            indent(file, "std::vector<std::string> vals{\"%s\"};" % ("\",\"".join(trimmedVals)))
            indent(file, "ncclCvarInit();")
            indent(file, "checkListValues<std::string>(vals, %s);" % (self.name))
            indent(file, "}")
            file.write("\n")

        indent(file, "TEST_F(CvarTest, %s_default_value) {" % (self.name))
        if self.default:
            indent(file, "testDefaultValue(\"%s\");" % (self.envstr))
            indent(file, "{")
            indent(file, "std::vector<std::string> vals{\"%s\"};" % (self.default.replace("," , "\",\"")))
            indent(file, "checkListValues<std::string>(vals, %s);" % (self.name))
            indent(file, "}")
        else:
            indent(file, "testDefaultValue(\"%s\");" % (self.envstr))
            indent(file, "EXPECT_EQ(%s.size(), 0);" % (self.name))
        indent(file, "}")
        file.write("\n")

        self.dupValUnitTest(file)

    def readenv(self, file):
        default = self.default if self.default else ""
        indent(file, "%s.clear();" % self.name)
        indent(file, "%s = env2strlist(\"%s\", \"%s\");" %
            (self.name, self.envstr, default))
        indent(file, "%s_DEFAULT.clear();" % self.name)
        indent(file, "%s_DEFAULT = env2strlist(\"NCCL_ENV_DO_NOT_SET\", \"%s\");" %
            (self.name, default))
        file.write("\n")


class prefixedStringlist(stringlist):
    @staticmethod
    def utilfns(file):
        pass

    def externDecl(self, file):
        indent(file, "extern std::string %s_PREFIX;" % self.name)
        indent(file, "extern std::string %s_PREFIX_DEFAULT;" % self.name)
        super().externDecl(file)

    def storageDecl(self, file):
        indent(file, "std::string %s_PREFIX;" % self.name)
        indent(file, "std::string %s_PREFIX_DEFAULT;" % self.name)
        super().storageDecl(file)

    def unitTest(self, file):
        indent(file, "TEST_F(CvarTest, %s_default_value) {" % (self.name))
        if self.default:
            indent(file, "testDefaultValue(\"%s\");" % (self.envstr))
            indent(file, "{")
            trimmedPrefixes = [v.strip() for v in self.prefixes.split(",")]
            default = self.default
            for v in trimmedPrefixes:
                default = default.lstrip(v)
            indent(file, "std::vector<std::string> vals{\"%s\"};" % (default.replace("," , "\",\"")))
            indent(file, "checkListValues<std::string>(vals, %s);" % (self.name))
            indent(file, "}")
        else:
            indent(file, "testDefaultValue(\"%s\");" % (self.envstr))
            indent(file, "EXPECT_EQ(%s.size(), 0);" % (self.name))
        indent(file, "}")
        file.write("\n")

        self.dupValUnitTest(file)

        val = "val1,val2,val3"
        for i, prefix in enumerate(["^", "=", ""]):
            indent(file, "TEST_F(CvarTest, %s_prefix_%s) {" % (self.name, i))
            indent(file, "setenv(\"%s\", \"%s%s\", 1);" % (self.envstr, prefix, val))
            indent(file, "std::vector<std::string> vals{\"%s\"};" % (val.replace("," , "\",\"")))
            indent(file, "ncclCvarInit();")
            indent(file, "EXPECT_EQ(%s_PREFIX, \"%s\");" % (self.name, prefix))
            indent(file, "checkListValues<std::string>(vals, %s);" % (self.name))
            indent(file, "}")
            file.write("\n")

    def readenv(self, file):
        trimmedPrefixes = [v.strip() for v in self.prefixes.split(",")]
        indent(file, "std::vector<std::string> %s_allPrefixes{\"%s\"};" % (self.name, ("\", \"").join(trimmedPrefixes)))
        default = self.default if self.default else ""
        indent(file, "%s.clear();" % self.name)
        indent(file, "std::tie(%s_PREFIX, %s) = env2prefixedStrlist(\"%s\", \"%s\", %s_allPrefixes);" %
                (self.name, self.name, self.envstr, default, self.name))
        indent(file, "%s_DEFAULT.clear();" % self.name)
        indent(file, "std::tie(%s_PREFIX_DEFAULT, %s_DEFAULT) = " \
                "env2prefixedStrlist(\"NCCL_ENV_DO_NOT_SET\", \"%s\", %s_allPrefixes);" %
                (self.name, self.name, default, self.name))
        file.write("\n")

class enum(basetype):
    @staticmethod
    def utilfns(file):
        pass

    def externDecl(self, file):
        choiceList = self.choices.replace(" ", "").split(",")
        indent(file, "enum class %s {" % self.name)
        for c in choiceList:
            indent(file, "%s," % c)
        indent(file, "};")
        indent(file, "extern enum %s %s;" % (self.name, self.name))
        indent(file, "extern enum %s %s_DEFAULT;" % (self.name, self.name))
        file.write("\n")

    def storageDecl(self, file):
        indent(file, "enum %s %s;" % (self.name, self.name))
        indent(file, "enum %s %s_DEFAULT;" % (self.name, self.name))

    def unitTest(self, file):
        choiceList = self.choices.replace(" ", "").split(",")
        for i, val in enumerate(choiceList):
            indent(file, "TEST_F(CvarTest, %s_single_choice_%s) {" % (self.name, i))
            indent(file, "setenv(\"%s\", \"%s\", 1);" % (self.envstr, val))
            indent(file, "ncclCvarInit();")
            indent(file, "EXPECT_EQ(%s, %s::%s);" % (self.name, self.name, val))
            indent(file, "}")
            file.write("\n")

        if self.default:
            indent(file, "TEST_F(CvarTest, %s_default_choice) {" % (self.name))
            indent(file, "testDefaultValue(\"%s\");" % (self.envstr))
            indent(file, "EXPECT_EQ(%s, %s::%s);" % (self.name, self.name, self.default))
            indent(file, "}")
        file.write("\n")

        self.unknownValUnitTest(file)

    def readenv(self, file):
        indent(file, "if (getenv(\"%s\") == nullptr) {" % self.envstr)
        indent(file, "%s = %s::%s;" % (self.name, self.name, self.default))
        indent(file, "} else {")
        indent(file, "std::string str(getenv(\"%s\"));" % self.envstr)
        choices = self.choices.replace(" ", "").split(",")
        for idx, c in enumerate(choices):
            if (idx == 0):
               indent(file, "if (str == std::string(\"%s\")) {" % c)
            else:
               indent(file, "} else if (str == std::string(\"%s\")) {" % c)
            indent(file, "%s = %s::%s;" % (self.name, self.name, c))
        indent(file, "} else {")
        indent(file, "  CVAR_WARN_UNKNOWN_VALUE(\"%s\", str.c_str());" % self.name)
        indent(file, "}")
        indent(file, "}")
        indent(file, "%s_DEFAULT = %s::%s;" % (self.name, self.name, self.default))
        file.write("\n")


class enumlist(basetype):
    @staticmethod
    def utilfns(file):
        pass

    def externDecl(self, file):
        choiceList = self.choices.replace(" ", "").split(",")
        indent(file, "enum class %s {" % self.name)
        for c in choiceList:
            indent(file, "%s," % c)
        indent(file, "};")
        indent(file, "extern std::vector<enum %s> %s;" % (self.name, self.name))
        indent(file, "extern std::vector<enum %s> %s_DEFAULT;" % (self.name, self.name))
        file.write("\n")

    def storageDecl(self, file):
        indent(file, "std::vector<enum %s> %s;" % (self.name, self.name))
        indent(file, "std::vector<enum %s> %s_DEFAULT;" % (self.name, self.name))

    def unitTest(self, file):
        choiceList = self.choices.replace(" ", "").split(",")
        allChoicesEnum = ["%s::%s" % (self.name, c) for c in choiceList]

        for i, val in enumerate(choiceList):
            indent(file, "TEST_F(CvarTest, %s_single_choice_%s) {" % (self.name, i))
            indent(file, "setenv(\"%s\", \"%s\", 1);" % (self.envstr, val))
            indent(file, "ncclCvarInit();")
            indent(file, "std::vector<enum %s> vals{%s::%s};" % (self.name, self.name, val))
            indent(file, "checkListValues<enum %s>(vals, %s);" % (self.name, self.name))
            indent(file, "}")
            file.write("\n")

        indent(file, "TEST_F(CvarTest, %s_all_choices) {" % (self.name))
        indent(file, "setenv(\"%s\", \"%s\", 1);" % (self.envstr, self.choices))
        indent(file, "ncclCvarInit();")
        indent(file, "std::vector<enum %s> vals{%s};" % (self.name, ",".join(allChoicesEnum)))
        indent(file, "checkListValues<enum %s>(vals, %s);" % (self.name, self.name))
        indent(file, "}")
        file.write("\n")

        defaultChoicesEnum = ["%s::%s" % (self.name, c) for c in self.default.replace(" ", "").split(",")]
        indent(file, "TEST_F(CvarTest, %s_default_choices) {" % (self.name))
        if defaultChoicesEnum:
            indent(file, "testDefaultValue(\"%s\");" % (self.envstr))
            indent(file, "std::vector<enum %s> vals{%s};" % (self.name, ",".join(defaultChoicesEnum)))
            indent(file, "checkListValues<enum %s>(vals, %s);" % (self.name, self.name))
        else:
            indent(file, "testDefaultValue(\"%s\");" % (self.envstr))
            indent(file, "EXPECT_EQ(%s.size(), 0);" % (self.name))
        indent(file, "}")
        file.write("\n")

        self.unknownValUnitTest(file)
        self.dupValUnitTest(file)

    def readenv(self, file):
        indent(file, "{")
        indent(file, "%s.clear();" % self.name)
        indent(file, "auto tokens = env2strlist(\"%s\", \"%s\");" %
                (self.envstr, self.default))
        choices = self.choices.replace(" ", "").split(",")
        indent(file, "for (auto token : tokens) {")
        for idx, c in enumerate(choices):
            if (idx == 0):
               indent(file, "if (token == std::string(\"%s\")) {" % c)
            else:
               indent(file, "} else if (token == std::string(\"%s\")) {" % c)
            indent(file, "%s.emplace_back(%s::%s);" % (self.name, self.name, c))
        indent(file, "} else {")
        indent(file, "  CVAR_WARN_UNKNOWN_VALUE(\"%s\", token.c_str());" % self.name)
        indent(file, "}")
        indent(file, "}")
        indent(file, "}")
        indent(file, "%s_DEFAULT.clear();" % self.name)
        default = self.default.replace(" ", "").split(",")
        for d in default:
            indent(file, "%s_DEFAULT.emplace_back(%s::%s);" % (self.name, self.name, d))
        file.write("\n")

def printAutogenHeader(file):
    file.write("// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.\n")
    file.write("\n")
    file.write("// Automatically generated by ./maint/extractcvars.py --- START\n")
    file.write("// DO NOT EDIT!!!\n")

def printAutogenFooter(file):
    file.write("// Automatically generated by ./maint/extractcvars.py --- END\n")

def populateCCFile(allcvars, outputFilename):
    file = open(outputFilename, "w")
    printAutogenHeader(file)
    file.write("\n")

    file.write("#include <iostream>\n")
    file.write("#include <unordered_set>\n")
    file.write("#include <string>\n")
    file.write("#include <vector>\n")
    file.write("#include \"nccl_cvars.h\"\n")
    file.write("#include \"nccl_cvars_base.h\"\n")
    file.write("\n")

    # Generate storage declaration
    for cvar in allcvars:
        cvar.storageDecl(file)
    file.write("\n")

    # Generate initialization for environment variable set
    indent(file, "void initEnvSet(std::unordered_set<std::string>& env) {")
    for cvar in allcvars:
        indent(file, "env.insert(\"%s\");" % cvar.envstr)
    for e in acceptedEnvs:
        indent(file, "env.insert(\"%s\");" % e)
    indent(file, "}")
    file.write("\n")

    # Generate environment reading of all cvars
    indent(file, "void readCvarEnv() {")
    for cvar in allcvars:
        cvar.readenv(file)
    indent(file, "}")
    file.write("\n")

    printAutogenFooter(file)
    file.close()


def populateHFile(allcvars, outputFilename):
    file = open(outputFilename, "w")
    printAutogenHeader(file)
    file.write("\n")

    file.write("#ifndef NCCL_CVARS_H_INCLUDED\n")
    file.write("#define NCCL_CVARS_H_INCLUDED\n")
    file.write("\n")

    file.write("#include <string>\n")
    file.write("#include <vector>\n")
    file.write("\n")

    # Generate extern declaration
    for cvar in allcvars:
        cvar.externDecl(file)
    file.write("\n")

    file.write("void ncclCvarInit();\n")
    file.write("\n")

    file.write("#endif  /* NCCL_CVARS_H_INCLUDED */")
    file.write("\n")

    printAutogenFooter(file)
    file.close()


def populateReadme(allcvars, filename):
    file = open(filename, "w")
    file.write("(c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.\n")
    file.write("\n")
    file.write("Automatically generated\n")
    file.write("  by ./maint/extractcvars.py\n")
    file.write("DO NOT EDIT!!!\n")
    for cvar in allcvars:
        cvar.desc(file)
    file.close()

def populateUT(allcvars, templateFilename, outputFilename):
    file = StringIO()

    # Generate unit test declarations
    file.write("// Automatically generated by ./maint/extractcvars.py\n")
    file.write("// DO NOT EDIT!!!\n")
    for cvar in allcvars:
        cvar.unitTest(file)
    utDecl = file.getvalue()

    # Load template and insert generated contents
    with open(templateFilename, "r") as tpl:
        fileContents = tpl.read()
        fileContents = fileContents.replace("## NCCL_CVAR_TESTS_DECL ##", utDecl)

        with open(outputFilename, "w") as out:
            out.write(fileContents)
    file.close()

def main():
    filenames = []
    for (root, dirs, files) in os.walk('.', topdown=True):
        for x in files:
            if x.endswith(".cc") or x.endswith(".h"):
                filenames.append(os.path.join(root, x))

    data = loadCvarsFromFiles(filenames)
    if (data['cvars'] == None):
        return

    loadedCvars = sorted(data['cvars'], key=lambda x: x['name'])

    allcvars = []
    for cvar in loadedCvars:
        if (cvar['type'] == "bool"):
            allcvars.append(bool(cvar))
        elif (cvar['type'] == "string"):
            allcvars.append(string(cvar))
        elif (cvar['type'] == "stringlist"):
            allcvars.append(stringlist(cvar))
        elif (cvar['type'] == "enum"):
            allcvars.append(enum(cvar))
        elif (cvar['type'] == "enumlist"):
            allcvars.append(enumlist(cvar))
        elif (cvar['type'] == "prefixed_stringlist"):
            allcvars.append(prefixedStringlist(cvar))
        else:
            allcvars.append(numeric(cvar))

    populateCCFile(allcvars, "src/misc/nccl_cvars.cc")
    populateHFile(allcvars, "src/include/nccl_cvars.h")
    populateReadme(allcvars, "README.cvars")
    populateUT(allcvars, "src/tests/CvarUT.cc.in", "src/tests/CvarUT.cc")


if __name__ == "__main__":
    main()
