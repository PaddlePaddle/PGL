import sys
sys.path.append(".")

def java_string_hashcode(s):
    """Mimic Java's hashCode in python 2"""
    try:
        s = unicode(s)
    except:
        try:
            s = unicode(s.decode('utf8'))
        except:
            raise Exception("Please enter a unicode type string or utf8 bytestring.")
    h = 1
    for c in s:
        h = int((((31 * h + ord(c)) ^ 0x80000000) & 0xFFFFFFFF) - 0x80000000)
    return h

def get_bits(m):
    count = 0
    while m > 0:
        m = m // 31
        count += 1
    return count + 1

def convert2str(m):
    t = ""
    #ori_m = m
    while m > 0:
        v = m % 31
        m = m // 31
        t = chr(v) + t 
    return t

def find_all_hashcode(m):
    hash_str = ["" ] * m
    boolist = [ False ] * m 
    
    start = 0
    done = 0
    while done < m:
        ss = convert2str(start)
        ss_set = ss
        if ('\t' not in ss_set) and ('\n' not in ss_set) and ('\r' not in ss_set) and len(ss_set) > 0:
            #hd = java_string_hashcode(ss)
            machine = java_string_hashcode(ss) % m
            if boolist[machine] == False:
                hash_str[machine] = convert2str(start) 
                boolist[machine] = True
                done += 1

        start += 1
    return hash_str

def mapper(num_machine, hash_code, symmetry='yes', node_type_shard='no'):
    # format: etype \t node_id \t others
    # format: src \t dst \t others
    for line in sys.stdin:  
        line = line.rstrip("\r\n")
        fields = line.split('\t')
        if node_type_shard == 'yes':
            machine = int(fields[1]) % num_machine
            sys.stdout.write("%s\t%s\n" % (hash_code[machine], line))
        else:
            machine = int(fields[0]) % num_machine
            sys.stdout.write("%s\t%s\n" % (hash_code[machine], line))

        if symmetry == 'yes':
            machine2 = int(fields[1]) % num_machine
            if machine2 != machine:
                sys.stdout.write("%s\t%s\n" % (hash_code[machine2], line))

def reducer():
    for line in sys.stdin:
        fields = line.rstrip("\r\n").split('\t')
        res = '\t'.join(fields[1:])
        sys.stdout.write("%s\n" % (res))

if __name__ == "__main__":
    if sys.argv[1] == "map":
        num_machine = int(sys.argv[2])
        symmetry = sys.argv[3]
        node_type_shard = sys.argv[4]
        hash_code = find_all_hashcode(num_machine)
        mapper(num_machine, hash_code, symmetry, node_type_shard)
    else:
        reducer()
