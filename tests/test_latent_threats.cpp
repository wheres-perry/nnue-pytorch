#include <iostream>
#include <cassert>
#include <vector>
#include <string>

#include "../data_loader/cpp/lib/chess.h"
#include "../data_loader/cpp/training_data_loader_internal.h"

using namespace chess;
using namespace binpack;

void test_slider_blockers_basic() {
    std::cout << "Testing slider_blockers basic scenarios..." << std::endl;

    // Position 1: Direct check (0 blockers)
    {
        Position pos = Position::fromFen("4k3/8/8/8/8/8/4R3/4K3 b - - 0 1");
        Bitboard pinners = Bitboard::none();
        Bitboard blockers = slider_blockers(pos, Color::White, pos.kingSquare(Color::Black), pinners);
        assert(blockers.count() == 0);
        assert(pinners.count() == 0);
    }

    // Position 2: White rook e2, White pawn e4, Black King e8 (1 blocker: e4)
    {
        Position pos = Position::fromFen("4k3/8/8/8/4P3/8/4R3/4K3 b - - 0 1");
        Bitboard pinners = Bitboard::none();
        Bitboard blockers = slider_blockers(pos, Color::White, pos.kingSquare(Color::Black), pinners);
        assert(blockers.count() == 1);
        assert(blockers.first() == e4);
        assert(pinners.count() == 1);
        assert(pinners.first() == e2);
    }

    // Position 3: Absolute pin: White King e1, White Pawn e5, Black Rook e7
    {
        Position pos = Position::fromFen("4k3/4r3/8/4P3/8/8/8/4K3 w - - 0 1");
        Bitboard pinners = Bitboard::none();
        Bitboard blockers = slider_blockers(pos, Color::Black, pos.kingSquare(Color::White), pinners);
        assert(blockers.count() == 1);
        assert(blockers.first() == e5);
        assert(pinners.count() == 1);
        assert(pinners.first() == e7);
    }

    // Position 4: Two blockers on same ray (not a pin)
    {
        Position pos = Position::fromFen("4k3/4r3/8/4P3/4N3/8/8/4K3 w - - 0 1");
        Bitboard pinners = Bitboard::none();
        Bitboard blockers = slider_blockers(pos, Color::Black, pos.kingSquare(Color::White), pinners);
        assert(blockers.count() == 0);
        assert(pinners.count() == 0);
    }

    // Position 5: Diagonal pin: Black Bishop c6, White Pawn e4, White King h1
    {
        Position pos = Position::fromFen("4k3/8/2b5/8/4P3/8/8/7K w - - 0 1");
        Bitboard pinners = Bitboard::none();
        Bitboard blockers = slider_blockers(pos, Color::Black, pos.kingSquare(Color::White), pinners);
        assert(blockers.count() == 1);
        assert(blockers.first() == e4);
        assert(pinners.count() == 1);
        assert(pinners.first() == c6);
    }

    // Position 6: Queen skewer: White Queen a1, Black Knight d4, Black King g7
    {
        Position pos = Position::fromFen("8/6k1/8/8/3n4/8/8/Q3K3 w - - 0 1");
        Bitboard pinners = Bitboard::none();
        Bitboard blockers = slider_blockers(pos, Color::White, pos.kingSquare(Color::Black), pinners);
        assert(blockers.count() == 1);
        assert(blockers.first() == d4);
        assert(pinners.count() == 1);
        assert(pinners.first() == a1);
    }

    std::cout << "slider_blockers basic tests passed!" << std::endl;
}

void test_feature_extractor() {
    std::cout << "Testing LatentThreats feature extraction and index bounds..." << std::endl;

    auto ext = get_feature("LatentThreats");
    assert(ext != nullptr);
    assert(ext->inputs() == 12288);
    assert(ext->max_active_features() == 8);

    // Test extraction on a rich position with both pins and skewers
    // White King g1, White Bishop b2, Black Pawn d4, Black King g7 (diagonal skewer by Bishop b2)
    // Black Rook g8 (pin on White Pawn g2 against White King g1!)
    {
        Position pos = Position::fromFen("6r1/6k1/8/8/3p4/8/1B4P1/6K1 w - - 0 1");
        TrainingDataEntry entry;
        entry.pos = pos;
        entry.score = 0;
        entry.result = 0;

        int features_w[8];
        for (int i = 0; i < 8; ++i) features_w[i] = -1;

        auto [count_w, inputs_w] = ext->fill_features_sparse(entry, features_w, Color::White);
        assert(inputs_w == 12288);
        assert(count_w >= 1 && count_w <= 8);

        for (int i = 0; i < count_w; ++i) {
            assert(features_w[i] >= 0 && features_w[i] < 12288);
            std::cout << "  Active feature [White pov] " << i << ": " << features_w[i] << std::endl;
        }

        int features_b[8];
        for (int i = 0; i < 8; ++i) features_b[i] = -1;

        auto [count_b, inputs_b] = ext->fill_features_sparse(entry, features_b, Color::Black);
        assert(inputs_b == 12288);
        assert(count_b >= 1 && count_b <= 8);

        for (int i = 0; i < count_b; ++i) {
            assert(features_b[i] >= 0 && features_b[i] < 12288);
            std::cout << "  Active feature [Black pov] " << i << ": " << features_b[i] << std::endl;
        }
    }

    // Test ComposedFeatureExtractor
    {
        auto composed = get_feature("Full_Threats+PP_3Wide+HalfKAv2_hm+LatentThreats");
        assert(composed != nullptr);
        assert(composed->inputs() == 59808 + 4560 + 24576 + 12288);
        assert(composed->max_active_features() == 128 + 128 + 32 + 8);
        std::cout << "Composed extractor total inputs: " << composed->inputs()
                  << ", max active: " << composed->max_active_features() << std::endl;
    }

    std::cout << "LatentThreats feature extractor tests passed!" << std::endl;
}

int main() {
    test_slider_blockers_basic();
    test_feature_extractor();
    std::cout << "All C++ tests passed successfully!" << std::endl;
    return 0;
}
