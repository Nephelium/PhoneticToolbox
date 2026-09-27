#!/usr/bin/env perl
use strict;
use warnings;
use Cwd qw(abs_path);
use Digest::SHA qw(sha256_hex);
use File::Basename qw(dirname basename);
use File::Path qw(make_path);
use IO::Socket::INET;
use JSON::PP;
use POSIX qw(strftime);

my $root = abs_path(dirname(__FILE__) . '/..');
my $dist = "$root/frontend/dist";
die "missing frontend build\n" unless -f "$dist/index.html";
my @targets = (
    'index.html',
    map { s/^\Q$dist\E\///r } (glob "$dist/assets/AppShell-*.js"),
    map { s/^\Q$dist\E\///r } (glob "$dist/assets/MandarinIpaPage-*.js"),
    map { s/^\Q$dist\E\///r } (glob "$dist/assets/MandarinIpaPage-*.css"),
    map { s/^\Q$dist\E\///r } (glob "$dist/assets/DoulosSIL-Regular-*.ttf"),
);
die "incomplete M13 static bundle\n" unless @targets == 5;

my $server = IO::Socket::INET->new(
    LocalAddr => '127.0.0.1', LocalPort => 0, Listen => 8, ReuseAddr => 1, Proto => 'tcp'
) or die "listen failed: $!\n";
my $port = $server->sockport;
my $pid = fork();
die "fork failed: $!\n" unless defined $pid;

if ($pid == 0) {
    for (1 .. scalar @targets) {
        my $client = $server->accept or exit 2;
        my $request = <$client> // '';
        $request =~ m{^GET\s+/(\S*)\s+HTTP/} or exit 3;
        my $relative = $1 eq '' ? 'index.html' : $1;
        exit 4 if $relative =~ m{\.\.} || $relative !~ m{\A[\w./-]+\z};
        while (my $line = <$client>) { last if $line =~ /^\r?\n$/ }
        my $file = "$dist/$relative";
        unless (-f $file) { print {$client} "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n"; close $client; next }
        open my $fh, '<:raw', $file or exit 5;
        local $/; my $body = <$fh>; close $fh;
        print {$client} "HTTP/1.1 200 OK\r\nContent-Length: " . length($body) . "\r\nConnection: close\r\n\r\n";
        print {$client} $body;
        close $client;
    }
    exit 0;
}
close $server;

my @files;
for my $relative (@targets) {
    my $client = IO::Socket::INET->new(PeerAddr => '127.0.0.1', PeerPort => $port, Proto => 'tcp')
        or die "connect failed: $!\n";
    print {$client} "GET /$relative HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
    local $/; my $response = <$client>; close $client;
    $response =~ m{\AHTTP/1\.1 200 OK\r\n.*?\r\n\r\n(.*)\z}s or die "failed response for $relative\n";
    my $body = $1;
    open my $fh, '<:raw', "$dist/$relative" or die $!;
    my $disk = <$fh>; close $fh;
    die "body mismatch for $relative\n" unless $body eq $disk;
    push @files, { path => $relative, bytes => length($body), sha256 => sha256_hex($body), status => 200 };
}
waitpid($pid, 0);
die "static server failed\n" unless $? == 0;

my $stamp = strftime('%Y%m%d-%H%M%S', localtime);
my $out = "$root/output/validation/m13-linux-static/$stamp";
make_path($out);
my $report = {
    success => JSON::PP::true,
    platform => scalar(`uname -srm`) =~ s/\s+\z//r,
    distro => scalar(`. /etc/os-release && printf '%s %s' "\$NAME" "\$VERSION_ID"`),
    server => 'IO::Socket::INET static HTTP on 127.0.0.1',
    compute_task => 'not used',
    external_network => 'not used',
    files => \@files,
};
open my $report_file, '>:encoding(UTF-8)', "$out/report.json" or die $!;
print {$report_file} JSON::PP->new->utf8(0)->pretty->canonical->encode($report);
close $report_file;
print "$out/report.json\n";
