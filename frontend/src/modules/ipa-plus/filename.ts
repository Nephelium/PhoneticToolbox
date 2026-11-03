/** Text export identity is independent of interface language. */
export function ipaTextFilename(now=new Date()):string{return '国际音标-'+now.toISOString().slice(0,10)+'.txt';}
